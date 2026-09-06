//! Echo control API definitions ported from `api/echo_control.h`.

/// Metrics exposed by an echo controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub echo_return_loss: f64,
    pub echo_return_loss_enhancement: f64,
    pub delay_ms: i32,
    pub nearend_active: bool,
    pub nearend_active_ratio: f64,
    pub echo_sum: f64,
    pub ne_sum: f64,
    pub noise_sum: f64,
    pub echo_to_nearend_ratio: f64,
    pub nearend_to_noise_ratio: f64,
    pub trigger_counter: i32,
    pub hold_counter: i32,
    pub initial_state: bool,
    pub enr_enter_margin: f64,
    pub snr_enter_margin: f64,
    pub exit_condition: bool,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            echo_return_loss: 0.0,
            echo_return_loss_enhancement: 0.0,
            delay_ms: 0,
            nearend_active: false,
            nearend_active_ratio: 0.0,
            echo_sum: 0.0,
            ne_sum: 0.0,
            noise_sum: 0.0,
            echo_to_nearend_ratio: 0.0,
            nearend_to_noise_ratio: 0.0,
            trigger_counter: 0,
            hold_counter: 0,
            initial_state: false,
            enr_enter_margin: 0.0,
            snr_enter_margin: 0.0,
            exit_condition: false,
        }
    }
}

/// Trait representing an acoustic echo cancellation submodule.
pub trait EchoControl {
    /// Buffer type used for render/capture processing.
    type Buffer;

    /// Analysis (without modification) of the render signal.
    fn analyze_render(&mut self, render: &mut Self::Buffer);

    /// Analysis (without modification) of the capture signal.
    fn analyze_capture(&mut self, capture: &mut Self::Buffer);

    /// Processes the capture signal in order to remove the echo.
    fn process_capture(&mut self, capture: &mut Self::Buffer, level_change: bool);

    /// Processes the capture signal and provides the linear filter output.
    fn process_capture_with_linear_output(
        &mut self,
        capture: &mut Self::Buffer,
        linear_output: &mut Self::Buffer,
        level_change: bool,
    );

    /// Collects current performance metrics.
    fn metrics(&self) -> Metrics;

    /// Provides an optional external estimate of the audio buffer delay.
    fn set_audio_buffer_delay(&mut self, delay_ms: i32);

    /// Whether the signal is altered.
    fn active_processing(&self) -> bool;
}

/// Trait for factories that create echo controllers.
pub trait EchoControlFactory {
    type Controller: EchoControl;

    fn create(
        &self,
        sample_rate_hz: i32,
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> Self::Controller;
}
