"""Visual-history GRU and reference-conditioned state sequence predictor."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class History:
    values: torch.Tensor              # [1,T,26], physical units
    valid: torch.Tensor               # [1,T,26], bool
    source_ns: torch.Tensor           # [1,T,26], original sensor timestamps
    available_ns: torch.Tensor        # [1,T,26], when the values became available
    ticks_ns: torch.Tensor            # [T], aligned observation ticks
    rgb: torch.Tensor                 # [1,T,3,64,96], uint8
    rgb_source_ns: torch.Tensor       # [T]
    rgb_available_ns: torch.Tensor    # [T]

    @property
    def decision_ns(self):
        return int(self.ticks_ns[-1])

    def tensors(self, device, *, max_age_ns, max_gap_ns):
        values = self.values.to(device, dtype=torch.float32)
        valid = self.valid.to(device)
        ticks = self.ticks_ns.to(device)
        source, available = self.source_ns.to(device), self.available_ns.to(device)
        t = len(ticks)
        if values.shape != (1, t, 26) or valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError('history requires values and boolean masks shaped [1,T,26]')
        if source.shape != values.shape or available.shape != values.shape or t < 2:
            raise ValueError('history timestamp shape mismatch')
        timestamp_arrays = (ticks, source, available, self.rgb_source_ns, self.rgb_available_ns)
        if any(v.dtype != torch.int64 for v in timestamp_arrays):
            raise ValueError('timestamps must be integer nanoseconds')
        gaps = ticks[1:]-ticks[:-1]
        if not bool(((gaps > 0) & (gaps <= max_gap_ns)).all()):
            raise ValueError('history time grid has gaps or is unordered')
        now = ticks.reshape(1, t, 1)
        if not bool((~valid | ((source <= available) & (available <= now))).all()):
            raise ValueError('future or incorrectly bound sensor observation')
        if not bool((~valid[..., 11:19] | (source[..., 11:19] < now)).all()):
            raise ValueError('actuator history must precede its observation tick')
        stale = valid & (now-source > max_age_ns)
        if not bool((valid & ~stale).any(dim=-1).all()) or not bool(torch.isfinite(values).all()):
            raise ValueError('empty or nonfinite history observation')
        if self.rgb.shape != (1, t, 3, 64, 96) or self.rgb.dtype != torch.uint8:
            raise ValueError('RGB input must be uint8 [1,T,3,64,96]')
        rgb_source, rgb_available = self.rgb_source_ns.to(device), self.rgb_available_ns.to(device)
        if rgb_source.shape != ticks.shape or rgb_available.shape != ticks.shape:
            raise ValueError('RGB timestamp grid mismatch')
        if not bool(((rgb_source <= rgb_available) & (rgb_available <= ticks) &
                     (ticks-rgb_source <= max_age_ns)).all()):
            raise ValueError('future or stale RGB history')
        age = torch.where(valid, (now-source).float()/1e9, torch.zeros_like(values))
        dt = torch.cat((gaps.new_zeros(1), gaps)).float().reshape(1, t, 1)/1e9
        return values, valid, stale, age, dt


class WorldModel(nn.Module):
    """Existing CNN + history GRU + conditional GRU computation, with direct tensor I/O.

    State[0:19] = DVL3, gyro3, acceleration3, pressure(kPa), altitude, actuators8.
    State[19:26] = anchor-relative quaternion XYZW and velocity XYZ.
    """
    def __init__(self, *, state_mean, state_scale, hidden_dim=64,
                 displacement_scale_m=1., velocity_scale_m_s=None,
                 reference_velocity_prior=False, max_age_ns=200_000_000,
                 max_gap_ns=100_000_000):
        super().__init__()
        mean = torch.as_tensor(state_mean, dtype=torch.float32)
        scale = torch.as_tensor(state_scale, dtype=torch.float32)
        if mean.shape != (19,) or scale.shape != (19,) or not bool(torch.isfinite(mean).all() & torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError('19 finite normalization means and positive scales required')
        if reference_velocity_prior and velocity_scale_m_s is None:
            raise ValueError('reference velocity prior requires the velocity-integral head')
        self.register_buffer('state_mean', mean.clone())
        self.register_buffer('state_scale', scale.clone())
        self.displacement_scale_m = displacement_scale_m
        self.velocity_scale_m_s = velocity_scale_m_s
        self.reference_velocity_prior = reference_velocity_prior
        self.max_age_ns, self.max_gap_ns = max_age_ns, max_gap_ns
        self.history_encoder = nn.Sequential(nn.Linear(4*26, hidden_dim), nn.ELU())
        self.history_gru = nn.GRU(hidden_dim+1, hidden_dim, batch_first=True)
        self.reference_encoder = nn.Sequential(nn.Linear(2*29, hidden_dim), nn.ELU())
        self.dynamics_gru = nn.GRU(hidden_dim+1, hidden_dim, batch_first=True)
        self.public_state_head = nn.Linear(hidden_dim, 19)
        if velocity_scale_m_s is None:
            self.displacement_head = nn.Linear(hidden_dim, 3)
        else:
            self.velocity_head = nn.Linear(hidden_dim, 3)
        self.visual_encoder = nn.Sequential(nn.Conv2d(3,16,3,2,1), nn.ELU(),
            nn.Conv2d(16,32,3,2,1), nn.ELU(), nn.Conv2d(32,16,3,2,1), nn.ELU())
        self.visual_project = nn.Sequential(nn.Flatten(2), nn.Linear(16*8*12,128), nn.ELU())
        self.visual_history = nn.GRU(129, hidden_dim, batch_first=True)
        self.visual_context = nn.Linear(hidden_dim, hidden_dim)
        self.future_visual_head = nn.Linear(hidden_dim,16*8*12)
        self.visual_decoder = nn.Sequential(nn.ConvTranspose2d(16,32,4,2,1), nn.ELU(),
            nn.ConvTranspose2d(32,16,4,2,1), nn.ELU(), nn.ConvTranspose2d(16,3,4,2,1), nn.Sigmoid())

    @classmethod
    def from_checkpoint(cls, path, *, device):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return model.eval().requires_grad_(False)

    def forward(self, history, references, *, decode_visual=False):
        device = self.state_mean.device
        values, hm, stale, age, hdt = history.tensors(device, max_age_ns=self.max_age_ns,
                                                    max_gap_ns=self.max_gap_ns)
        ids = tuple(r.candidate_id for r in references)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('distinct candidate identities required')
        import numpy as np
        times = references[0].times_s
        if any(not np.array_equal(r.times_s, times) for r in references):
            raise ValueError('candidate reference grids differ')
        encoded_references = [r.conditions() for r in references]
        conditions = torch.as_tensor(np.stack([r[0] for r in encoded_references]), device=device)
        cm = torch.as_tensor(np.stack([r[1] for r in encoded_references]), device=device)
        dt = torch.as_tensor(np.diff(times), dtype=torch.float32, device=device).reshape(1,-1,1).expand(len(ids),-1,-1)
        fresh = hm & ~stale
        state = torch.where(fresh[..., :19], values[..., :19], self.state_mean)
        normalized = torch.cat(((state-self.state_mean)/self.state_scale,
            torch.where(fresh[..., 19:], values[..., 19:], torch.zeros_like(values[..., 19:]))), dim=-1)
        hfeatures = self.history_encoder(torch.cat((normalized, hm.float(), stale.float(), age), dim=-1))
        _, h = self.history_gru(torch.cat((hfeatures, hdt), dim=-1))
        pixels = history.rgb.to(device, dtype=torch.float32)/255.
        b, t = pixels.shape[:2]
        encoded = self.visual_encoder(pixels.reshape(-1,3,64,96)).reshape(b,t,16,8,12)
        embedding = self.visual_project(encoded)
        _, visual_h = self.visual_history(torch.cat((embedding,hdt),dim=-1))
        h = h + self.visual_context(visual_h)
        cm[...,25:28] = False
        c = torch.where(cm, conditions, torch.zeros_like(conditions))
        features = self.reference_encoder(torch.cat((c,cm.float()),dim=-1))
        features, _ = self.dynamics_gru(torch.cat((features,dt),dim=-1), h.expand(-1,len(ids),-1).contiguous())
        current = torch.where(fresh[:,-1:,:19], values[:,-1:,:19], self.state_mean)
        states = self.public_state_head(features)*self.state_scale+current
        if self.velocity_scale_m_s is None:
            displacement = self.displacement_head(features)*self.displacement_scale_m
        else:
            velocity = self.velocity_head(features)*self.velocity_scale_m_s
            if self.reference_velocity_prior:
                velocity = velocity+c[...,7:10]
            displacement = torch.cumsum(velocity*dt,dim=1)
        result = dict(candidate_ids=ids, decision_ns=history.decision_ns,
                      public_state_sequence=states, displacement_sequence=displacement)
        if decode_visual:
            n,k = features.shape[:2]
            z = self.future_visual_head(features).reshape(n,k,16,8,12)
            result['future_rgb'] = self.visual_decoder(z.reshape(-1,16,8,12)).reshape(n,k,3,64,96)
        return result
