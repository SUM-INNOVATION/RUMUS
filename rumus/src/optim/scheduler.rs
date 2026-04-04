//! Learning rate schedulers.
//!
//! Schedulers compute the learning rate; the user applies it via
//! `optimizer.set_lr(scheduler.get_lr())`.  Schedulers do NOT own
//! the optimizer — this avoids borrow conflicts during training.

/// Trait for learning rate schedulers.
pub trait LRScheduler {
    /// Advance the scheduler by one step (typically called once per epoch).
    fn step(&mut self);

    /// Current learning rate.
    fn get_lr(&self) -> f32;

    /// Override the initial learning rate (e.g., for warmup).
    fn set_initial_lr(&mut self, lr: f32);
}

/// Step-decay scheduler: multiply LR by `gamma` every `step_size` epochs.
///
/// `lr = initial_lr * gamma^(epoch / step_size)`
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_epoch: usize,
    current_lr: f32,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        assert!(step_size > 0, "StepLR: step_size must be > 0");
        Self {
            initial_lr,
            step_size,
            gamma,
            current_epoch: 0,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self) {
        self.current_epoch += 1;
        if self.current_epoch % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn set_initial_lr(&mut self, lr: f32) {
        self.initial_lr = lr;
        self.current_lr = lr * self.gamma.powi((self.current_epoch / self.step_size) as i32);
    }
}

/// Cosine annealing scheduler: smooth decay from `initial_lr` to `eta_min`
/// over `t_max` epochs.
///
/// `lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * epoch / t_max))`
///
/// After `t_max` epochs, LR clamps at `eta_min` (no warm restarts).
pub struct CosineAnnealingLR {
    initial_lr: f32,
    t_max: usize,
    eta_min: f32,
    current_epoch: usize,
    current_lr: f32,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: usize, eta_min: f32) -> Self {
        assert!(t_max > 0, "CosineAnnealingLR: t_max must be > 0");
        Self {
            initial_lr,
            t_max,
            eta_min,
            current_epoch: 0,
            current_lr: initial_lr,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self) {
        self.current_epoch += 1;
        if self.current_epoch >= self.t_max {
            self.current_lr = self.eta_min;
        } else {
            self.current_lr = self.eta_min
                + 0.5
                    * (self.initial_lr - self.eta_min)
                    * (1.0
                        + (std::f32::consts::PI * self.current_epoch as f32
                            / self.t_max as f32)
                            .cos());
        }
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn set_initial_lr(&mut self, lr: f32) {
        self.initial_lr = lr;
        // Recompute current LR from the new base.
        if self.current_epoch >= self.t_max {
            self.current_lr = self.eta_min;
        } else {
            self.current_lr = self.eta_min
                + 0.5
                    * (lr - self.eta_min)
                    * (1.0
                        + (std::f32::consts::PI * self.current_epoch as f32
                            / self.t_max as f32)
                            .cos());
        }
    }
}
