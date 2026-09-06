use crate::audio_processing::aec3::aec3_common::FFT_LENGTH_BY_2_PLUS_1;
use crate::audio_processing::aec3::dominant_nearend_detector::DominantNearendDiagnostics;

pub trait NearendDetector: Send {
    fn is_nearend_state(&self) -> bool;
    fn diagnostics(&self) -> DominantNearendDiagnostics {
        DominantNearendDiagnostics::default()
    }
    fn update(
        &mut self,
        nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        initial_state: bool,
    );
}

