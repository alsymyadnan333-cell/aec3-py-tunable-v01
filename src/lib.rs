use aec3::api::config::EchoCanceller3Config;
use aec3::api::control::Metrics as RustMetrics;
use aec3::voip::{VoipAec3, VoipAec3Builder, VoipAec3Error};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-facing metrics object: thin wrapper around aec3::api::control::Metrics
#[pyclass(name = "Metrics")]
#[derive(Clone, Debug)]
pub struct PyMetrics {
    /// Echo Return Loss (dB)
    #[pyo3(get)]
    pub echo_return_loss: f64,
    /// Echo Return Loss Enhancement (dB)
    #[pyo3(get)]
    pub echo_return_loss_enhancement: f64,
    /// Estimated delay (ms)
    #[pyo3(get)]
    pub delay_ms: i32,
    /// Near-end speech active flag for the latest frame
    #[pyo3(get)]
    pub nearend_active: bool,
    /// Cumulative near-end speech active ratio across all processed frames
    #[pyo3(get)]
    pub nearend_active_ratio: f64,
}

impl From<RustMetrics> for PyMetrics {
    fn from(m: RustMetrics) -> Self {
        PyMetrics {
            echo_return_loss: m.echo_return_loss,
            echo_return_loss_enhancement: m.echo_return_loss_enhancement,
            delay_ms: m.delay_ms,
            nearend_active: m.nearend_active,
            nearend_active_ratio: m.nearend_active_ratio,
        }
    }
}

/// High-level wrapper around aec3::voip::VoipAec3, using NumPy arrays.
#[pyclass(name = "Aec3", unsendable)]
pub struct PyAec3 {
    inner: VoipAec3,
    frame_samples: usize,
    render_channels: usize,
    capture_channels: usize,
    #[pyo3(get)]
    dominant_nearend_enr_threshold: f32,
    #[pyo3(get)]
    dominant_nearend_snr_threshold: f32,
    #[pyo3(get)]
    dominant_nearend_trigger_threshold: usize,
    #[pyo3(get)]
    dominant_nearend_hold_duration: usize,
    #[pyo3(get)]
    dominant_nearend_use_during_initial_phase: bool,
    #[pyo3(get)]
    dominant_nearend_enr_exit_threshold: f32,
    #[pyo3(get)]
    nearend_enr_transparent: f32,
}

fn map_voip_err(err: VoipAec3Error) -> PyErr {
    PyErr::new::<PyValueError, _>(err.to_string())
}

fn bad_len(kind: &str, got: usize, expected: usize) -> PyErr {
    PyErr::new::<PyValueError, _>(format!(
        "{kind} length {got} != expected {expected} (frame_samples * {kind}_channels)"
    ))
}

fn not_contiguous(kind: &str, e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<PyValueError, _>(format!(
        "{kind} array must be contiguous in memory: {e}"
    ))
}

#[pymethods]
impl PyAec3 {
    #[new]
    #[pyo3(
        signature = (
            sample_rate_hz,
            render_channels,
            capture_channels,
            initial_delay_ms = None,
            enable_high_pass = None,
            dominant_nearend_enr_threshold = None,
            dominant_nearend_snr_threshold = None,
            dominant_nearend_trigger_threshold = None,
            dominant_nearend_hold_duration = None,
            dominant_nearend_use_during_initial_phase = None,
            dominant_nearend_enr_exit_threshold = None,
            nearend_enr_transparent = None,
        )
    )]
    fn new(
        sample_rate_hz: i32,
        render_channels: usize,
        capture_channels: usize,
        initial_delay_ms: Option<i32>,
        enable_high_pass: Option<bool>,
        dominant_nearend_enr_threshold: Option<f32>,
        dominant_nearend_snr_threshold: Option<f32>,
        dominant_nearend_trigger_threshold: Option<usize>,
        dominant_nearend_hold_duration: Option<usize>,
        dominant_nearend_use_during_initial_phase: Option<bool>,
        dominant_nearend_enr_exit_threshold: Option<f32>,
        nearend_enr_transparent: Option<f32>,
    ) -> PyResult<Self> {
        let mut config = EchoCanceller3Config::default();

        if let Some(val) = dominant_nearend_enr_threshold {
            if !val.is_finite() || val <= 0.0 || val > 100.0 {
                return Err(PyValueError::new_err(
                    "dominant_nearend_enr_threshold must be a finite, positive float between 0.0 and 100.0",
                ));
            }
            config.suppressor.dominant_nearend_detection.enr_threshold = val;
        }

        if let Some(val) = dominant_nearend_snr_threshold {
            if !val.is_finite() || val <= 0.0 || val > 1000.0 {
                return Err(PyValueError::new_err(
                    "dominant_nearend_snr_threshold must be a finite, positive float between 0.0 and 1000.0",
                ));
            }
            config.suppressor.dominant_nearend_detection.snr_threshold = val;
        }

        if let Some(val) = dominant_nearend_trigger_threshold {
            if val < 1 || val > 500 {
                return Err(PyValueError::new_err(
                    "dominant_nearend_trigger_threshold must be between 1 and 500 blocks",
                ));
            }
            config.suppressor.dominant_nearend_detection.trigger_threshold = val;
        }

        if let Some(val) = dominant_nearend_hold_duration {
            if val < 1 || val > 5000 {
                return Err(PyValueError::new_err(
                    "dominant_nearend_hold_duration must be between 1 and 5000 blocks",
                ));
            }
            config.suppressor.dominant_nearend_detection.hold_duration = val;
        }

        if let Some(val) = dominant_nearend_use_during_initial_phase {
            config.suppressor.dominant_nearend_detection.use_during_initial_phase = val;
        }

        if let Some(val) = dominant_nearend_enr_exit_threshold {
            if !val.is_finite() || val <= 0.0 || val > 100.0 {
                return Err(PyValueError::new_err(
                    "dominant_nearend_enr_exit_threshold must be a finite, positive float between 0.0 and 100.0",
                ));
            }
            config.suppressor.dominant_nearend_detection.enr_exit_threshold = val;
        }

        if let Some(trans) = nearend_enr_transparent {
            if !trans.is_finite() || trans < 0.0 || trans > 100.0 {
                return Err(PyValueError::new_err(
                    "nearend_enr_transparent must be a finite, non-negative float between 0.0 and 100.0",
                ));
            }
            config.suppressor.nearend_tuning.mask_lf.enr_transparent = trans;
            if config.suppressor.nearend_tuning.mask_lf.enr_suppress <= trans {
                config.suppressor.nearend_tuning.mask_lf.enr_suppress = trans + 0.01;
            }
        }

        let rec_enr = config.suppressor.dominant_nearend_detection.enr_threshold;
        let rec_snr = config.suppressor.dominant_nearend_detection.snr_threshold;
        let rec_trig = config.suppressor.dominant_nearend_detection.trigger_threshold;
        let rec_hold = config.suppressor.dominant_nearend_detection.hold_duration;
        let rec_init = config.suppressor.dominant_nearend_detection.use_during_initial_phase;
        let rec_exit = config.suppressor.dominant_nearend_detection.enr_exit_threshold;
        let rec_trans = config.suppressor.nearend_tuning.mask_lf.enr_transparent;

        let mut builder: VoipAec3Builder =
            VoipAec3::builder(sample_rate_hz, render_channels, capture_channels)
                .with_config(config);

        if let Some(delay) = initial_delay_ms {
            builder = builder.initial_delay_ms(delay);
        }
        if let Some(hp) = enable_high_pass {
            builder = builder.enable_high_pass(hp);
        }

        let pipeline = builder.build().map_err(map_voip_err)?;
        let frame_samples = pipeline.frame_samples();

        Ok(Self {
            inner: pipeline,
            frame_samples,
            render_channels,
            capture_channels,
            dominant_nearend_enr_threshold: rec_enr,
            dominant_nearend_snr_threshold: rec_snr,
            dominant_nearend_trigger_threshold: rec_trig,
            dominant_nearend_hold_duration: rec_hold,
            dominant_nearend_use_during_initial_phase: rec_init,
            dominant_nearend_enr_exit_threshold: rec_exit,
            nearend_enr_transparent: rec_trans,
        })
    }

    #[getter]
    fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    #[getter]
    fn sample_rate_hz(&self) -> i32 {
        self.inner.sample_rate_hz()
    }

    #[getter]
    fn is_nearend_active(&self) -> bool {
        self.inner.is_nearend_active()
    }

    #[getter]
    fn nearend_active_frames(&self) -> usize {
        self.inner.nearend_active_frames()
    }

    #[getter]
    fn total_frames(&self) -> usize {
        self.inner.total_frames()
    }

    #[getter]
    fn nearend_active_ratio(&self) -> f64 {
        self.inner.nearend_active_ratio()
    }

    fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        self.inner.set_audio_buffer_delay(delay_ms);
    }

    fn metrics(&self) -> PyMetrics {
        PyMetrics::from(self.inner.metrics())
    }

    fn handle_render_frame(&mut self, render_frame: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let slice = render_frame
            .as_slice()
            .map_err(|e| not_contiguous("render_frame", e))?;

        let expected = self.frame_samples * self.render_channels;
        if slice.len() != expected {
            return Err(bad_len("render_frame", slice.len(), expected));
        }

        self.inner.handle_render_frame(slice).map_err(map_voip_err)
    }

    #[pyo3(signature = (capture_frame, level_change=false))]
    fn process_capture_frame<'py>(
        &mut self,
        py: Python<'py>,
        capture_frame: PyReadonlyArray1<'py, f32>,
        level_change: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, PyMetrics)> {
        let capture_slice = capture_frame
            .as_slice()
            .map_err(|e| not_contiguous("capture_frame", e))?;

        let expected = self.frame_samples * self.capture_channels;
        if capture_slice.len() != expected {
            return Err(bad_len("capture_frame", capture_slice.len(), expected));
        }

        let mut out = vec![0.0f32; capture_slice.len()];
        let metrics = self
            .inner
            .process_capture_frame(capture_slice, level_change, &mut out)
            .map_err(map_voip_err)?;

        let out_array = out.into_pyarray(py);
        Ok((out_array, PyMetrics::from(metrics)))
    }

    #[pyo3(signature = (capture_frame, render_frame=None, level_change=false))]
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        capture_frame: PyReadonlyArray1<'py, f32>,
        render_frame: Option<PyReadonlyArray1<'py, f32>>,
        level_change: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, PyMetrics)> {
        let capture_slice = capture_frame
            .as_slice()
            .map_err(|e| not_contiguous("capture_frame", e))?;

        let expected_capture = self.frame_samples * self.capture_channels;
        if capture_slice.len() != expected_capture {
            return Err(bad_len("capture_frame", capture_slice.len(), expected_capture));
        }

        let render_slice_opt: Option<&[f32]> = if let Some(ref arr) = render_frame {
            let slice = arr
                .as_slice()
                .map_err(|e| not_contiguous("render_frame", e))?;

            let expected_render = self.frame_samples * self.render_channels;
            if slice.len() != expected_render {
                return Err(bad_len("render_frame", slice.len(), expected_render));
            }

            Some(slice)
        } else {
            None
        };

        let mut out = vec![0.0f32; capture_slice.len()];
        let metrics = self
            .inner
            .process(capture_slice, render_slice_opt, level_change, &mut out)
            .map_err(map_voip_err)?;

        let out_array = out.into_pyarray(py);
        Ok((out_array, PyMetrics::from(metrics)))
    }
}

#[pymodule]
fn aec3_py_tunable(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAec3>()?;
    m.add_class::<PyMetrics>()?;
    Ok(())
}
