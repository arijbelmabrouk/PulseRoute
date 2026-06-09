import { useWebSocket } from '../hooks/useWebSocket'
import { useEffect, useState } from 'react'

const C = {
  blue:      '#0072CE',
  blueDark:  '#005BA4',
  blueLight: '#E8F4FF',
  navy:      '#1A2B4A',
  bodyText:  '#3D4F6B',
  muted:     '#6B7A91',
  border:    '#D8E6F3',
  borderLt:  '#EDF4FB',
  bg:        '#F4F8FC',
  white:     '#FFFFFF',
  green:     '#00A86B',
  greenBg:   '#E6F7F1',
  amber:     '#E07B00',
  amberBg:   '#FFF4E0',
  rose:      '#C0392B',
  roseBg:    '#FCECEA',
  sans:      "'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
}

const STEPS_FACE = [
  { id: 1,  label: 'Camera initialised'           },
  { id: 2,  label: 'Face detected and calibrated' },
  { id: 3,  label: 'Signal captured'              },
  { id: 5,  label: 'Pulse waveform extracted'     },
  { id: 6,  label: 'Signal filtered'              },
  { id: 9,  label: 'Heart rate computed'          },
  { id: 10, label: 'Respiratory rate computed'    },
  { id: 11, label: 'Quality assessment complete'  },
]

const STEPS_PALM = [
  { id: 2,  label: 'Palm detected and calibrated' },
  { id: 3,  label: 'Palm signal captured'         },
  { id: 5,  label: 'Pulse waveform extracted'     },
  { id: 6,  label: 'Signal filtered'              },
  { id: 9,  label: 'Heart rate computed'          },
  { id: 10, label: 'Respiratory rate computed'    },
  { id: 11, label: 'Quality assessment complete'  },
]

function StepRow({ label, done, active, last }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 14,
      padding: '10px 0',
      borderBottom: last ? 'none' : `1px solid ${C.borderLt}`,
      opacity: done || active ? 1 : 0.35,
      transition: 'opacity 0.4s ease',
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: 8, flexShrink: 0,
        background: done ? C.blue : active ? C.blueLight : C.bg,
        border: `1.5px solid ${done ? C.blue : active ? C.blue : C.border}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        transition: 'all 0.35s ease',
        boxShadow: done ? `0 2px 8px ${C.blue}44` : 'none',
      }}>
        {done ? (
          <svg width="12" height="9" viewBox="0 0 12 9" fill="none">
            <path d="M1 4.5l3.5 3.5 6.5-7"
              stroke={C.white} strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        ) : active ? (
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: C.blue, animation: 'pulse 1.4s ease infinite',
          }} />
        ) : (
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: C.border }} />
        )}
      </div>
      <span style={{
        fontSize: 13, fontFamily: C.sans,
        color: done ? C.navy : active ? C.blue : C.muted,
        fontWeight: done ? 600 : active ? 500 : 400,
        flex: 1,
      }}>{label}</span>
      {done && <span style={{ fontSize: 10, fontWeight: 700, color: C.green, letterSpacing: '0.07em', textTransform: 'uppercase' }}>Done</span>}
      {active && !done && <span style={{ fontSize: 10, fontWeight: 700, color: C.blue, letterSpacing: '0.07em', textTransform: 'uppercase', animation: 'fadeInOut 1.5s ease infinite' }}>Active</span>}
    </div>
  )
}

function ResultCard({ label, value, unit, accent, delay = 0 }) {
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    if (value != null) {
      const t = setTimeout(() => setVisible(true), delay)
      return () => clearTimeout(t)
    }
  }, [value != null])
  return (
    <div style={{
      background: accent ? C.blue : C.white,
      border: `1.5px solid ${accent ? C.blue : C.border}`,
      borderRadius: 16, padding: '1.5rem 1rem', textAlign: 'center',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0) scale(1)' : 'translateY(12px) scale(0.97)',
      transition: 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)',
      boxShadow: accent ? `0 8px 24px ${C.blue}33` : '0 2px 8px rgba(0,0,0,0.06)',
    }}>
      <div style={{ fontSize: '2.4rem', fontWeight: 700, lineHeight: 1, color: accent ? C.white : C.navy, letterSpacing: '-0.03em', marginBottom: 6 }}>{value ?? '—'}</div>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: accent ? 'rgba(255,255,255,0.75)' : C.muted }}>{label}</div>
      {unit && <div style={{ fontSize: 11, color: accent ? 'rgba(255,255,255,0.6)' : C.muted, marginTop: 2 }}>{unit}</div>}
    </div>
  )
}

function ModeOption({ id, selected, onSelect, title, description }) {
  const isSelected = selected === id
  return (
    <button onClick={() => onSelect(id)} style={{
      display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
      gap: 5, padding: '14px 16px', borderRadius: 12, cursor: 'pointer',
      textAlign: 'left', width: '100%',
      background: isSelected ? C.blueLight : C.white,
      border: `1.5px solid ${isSelected ? C.blue : C.border}`,
      transition: 'all 0.2s ease',
      boxShadow: isSelected ? `0 0 0 3px ${C.blue}22` : 'none',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{
          width: 16, height: 16, borderRadius: '50%',
          border: `2px solid ${isSelected ? C.blue : C.border}`,
          background: isSelected ? C.blue : C.white,
          flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          {isSelected && <div style={{ width: 6, height: 6, borderRadius: '50%', background: C.white }} />}
        </div>
        <span style={{ fontSize: 14, fontWeight: 700, color: isSelected ? C.blue : C.navy }}>{title}</span>
      </div>
      <p style={{ fontSize: 12, color: C.muted, lineHeight: 1.5, paddingLeft: 24 }}>{description}</p>
    </button>
  )
}

function PreMeasurementForm({ onSubmit }) {
  const [name, setName]             = useState('')
  const [mode, setMode]             = useState('auto')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError]           = useState('')

  const handleSubmit = async () => {
    if (!name.trim()) { setError('Please enter your name or patient ID.'); return }
    setError('')
    setSubmitting(true)
    try {
      await fetch('http://localhost:8000/api/patient-id', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: name.trim(), mode }),
      })
      onSubmit(name.trim(), mode)
    } catch {
      setError('Could not connect to the server. Please try again.')
      setSubmitting(false)
    }
  }

  return (
    <div style={{ padding: '1.75rem' }}>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: C.navy, marginBottom: '0.35rem' }}>Before we begin</h2>
      <p style={{ fontSize: 13, color: C.muted, marginBottom: '1.5rem', lineHeight: 1.5 }}>
        Please enter your details so your doctor can identify your results.
      </p>
      <div style={{ marginBottom: '1.25rem' }}>
        <label style={{ display: 'block', fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', color: C.muted, marginBottom: 6 }}>Your name or patient ID</label>
        <input
          value={name} onChange={e => setName(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          placeholder="e.g. Jane Smith or P-00412"
          style={{ width: '100%', padding: '10px 14px', borderRadius: 10, fontSize: 14, border: `1.5px solid ${error ? C.rose : C.border}`, color: C.navy, outline: 'none', background: C.white }}
          onFocus={e => e.target.style.borderColor = C.blue}
          onBlur={e => e.target.style.borderColor = error ? C.rose : C.border}
        />
        {error && <p style={{ fontSize: 11, color: C.rose, marginTop: 5 }}>{error}</p>}
      </div>
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', color: C.muted, marginBottom: 8 }}>Measurement mode</label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <ModeOption id="auto" selected={mode} onSelect={setMode} title="Auto"
            description="Face first, automatically switches to palm if the signal is too weak." />
          <ModeOption id="palm" selected={mode} onSelect={setMode} title="Palm"
            description="Recommended if you have dark skin, a heavy beard, or face measurement has failed before." />
        </div>
      </div>
      <button onClick={handleSubmit} disabled={submitting} style={{
        width: '100%', padding: '12px', borderRadius: 12, fontSize: 14, fontWeight: 700,
        border: 'none', cursor: submitting ? 'default' : 'pointer',
        background: submitting ? C.muted : C.blue, color: C.white,
        boxShadow: submitting ? 'none' : `0 4px 16px ${C.blue}44`,
      }}>
        {submitting ? 'Connecting…' : "I'm ready"}
      </button>
    </div>
  )
}

// ── Main ─────────────────────────────────────────────

export default function PatientPage() {
  const { state, connected, latestFrame } = useWebSocket('/ws/patient')

  const [submitted,    setSubmitted]    = useState(false)
  const [patientName,  setPatientName]  = useState('')
  const [selectedMode, setSelectedMode] = useState('auto')

  const handleFormSubmit = (name, mode) => {
    setPatientName(name)
    setSelectedMode(mode)
    setSubmitted(true)
  }

  const isFailed   = state.status === 'failed'
  const isIdle     = !state.status || state.status === 'idle'
  const isRunning  = state.status === 'running' || state.status === 'starting'
  const isComplete = state.status === 'complete'
  const isPalmLive = state.modality === 'palm' || state.route_palm === true || selectedMode === 'palm'

  const final     = state.final || {}
  const steps     = state.steps || {}
  const doneSteps = Object.keys(steps).map(Number)
  const STEPS     = isPalmLive ? STEPS_PALM : STEPS_FACE
  const pct       = Math.max(0, Math.min(100, Number(state.progress || 0)))

  // Show live feed as soon as we have a frame — Step 2 or Step 3
  const pipelineActive = isRunning || state.status === 'starting'
  const showFeed = pipelineActive && latestFrame != null

  const statusColor = isFailed ? C.rose : isComplete ? C.green : isRunning ? C.blue : C.muted

  return (
    <div style={{ minHeight: '100vh', background: C.bg, fontFamily: C.sans, color: C.navy, display: 'flex', flexDirection: 'column' }}>

      {/* Nav */}
      <nav style={{
        background: C.white, borderBottom: `1px solid ${C.border}`,
        padding: '0 2rem', height: 60,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        position: 'sticky', top: 0, zIndex: 20,
        boxShadow: '0 1px 4px rgba(0,114,206,0.06)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
            <circle cx="17" cy="17" r="16" stroke={C.blue} strokeWidth="2.5" fill="none"/>
            <circle cx="17" cy="17" r="8"  stroke={C.blue} strokeWidth="2"   fill="none"/>
            <circle cx="17" cy="17" r="3"  fill={C.blue}/>
          </svg>
          <div>
            <span style={{ fontWeight: 700, fontSize: 16, color: C.navy }}>PulseRoute</span>
            <span style={{ display: 'block', fontSize: 10, color: C.muted, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Patient Monitoring</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {patientName && (
            <span style={{ fontSize: 12, color: C.navy, fontWeight: 500, padding: '4px 12px', borderRadius: 20, background: C.blueLight, border: `1px solid ${C.border}` }}>{patientName}</span>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 12, fontWeight: 500, color: connected ? C.green : C.muted, background: connected ? C.greenBg : C.bg, padding: '5px 14px', borderRadius: 20, border: `1px solid ${connected ? C.green + '44' : C.border}` }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: connected ? C.green : C.muted, animation: connected ? 'pulse 2s ease infinite' : 'none' }} />
            {connected ? 'Connected' : 'Reconnecting'}
          </div>
        </div>
      </nav>

      {/* Content */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'flex-start', justifyContent: 'center', padding: '2rem 1rem' }}>
        <div style={{ width: '100%', maxWidth: 520 }}>

          <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
            <h1 style={{ fontSize: '1.5rem', fontWeight: 700, color: C.navy, marginBottom: '0.4rem', letterSpacing: '-0.025em' }}>Contactless Vital Signs</h1>
            <p style={{ fontSize: 13, color: C.muted }}>Remote measurement via camera — no contact required</p>
          </div>

          <div style={{
            background: C.white,
            border: `1.5px solid ${isFailed ? C.rose + '55' : C.border}`,
            borderRadius: 20, overflow: 'hidden',
            boxShadow: '0 4px 20px rgba(0,114,206,0.08)',
          }}>

            {/* Pre-measurement form */}
            {isIdle && !submitted && <PreMeasurementForm onSubmit={handleFormSubmit} />}

            {/* ── LIVE FEED ──────────────────────────────────────
                The img tag shows exactly what the OpenCV window
                shows. The HUD text (Phase 1/2 labels, countdown,
                ITA, mask overlay colours, motion warnings,
                recording progress bar) is all drawn directly onto
                the frame pixels by _draw_setup_hud / run_signal_
                extraction before on_frame encodes it to JPEG.
                No React overlays — nothing is added on top.      */}
            {showFeed && (
              <img
                src={latestFrame}
                alt="Live camera feed"
                style={{
                  width: '100%',
                  aspectRatio: '4/3',
                  objectFit: 'cover',
                  display: 'block',
                  borderBottom: `1px solid ${C.border}`,
                }}
              />
            )}

            {/* Status header — shown when no frame yet or complete/failed */}
            {!showFeed && (submitted || isRunning || isComplete || isFailed) && (
              <div style={{
                padding: '2rem',
                background: isFailed ? C.roseBg : isComplete ? C.greenBg : isRunning ? C.blueLight : C.bg,
                borderBottom: `1px solid ${C.border}`,
                display: 'flex', alignItems: 'flex-start', gap: '1.25rem',
              }}>
                <div style={{ width: 12, height: 12, borderRadius: '50%', background: statusColor, flexShrink: 0, marginTop: 5, animation: isRunning ? 'pulse 1.4s ease infinite' : 'none' }} />
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'inline-block', fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', padding: '3px 10px', borderRadius: 20, marginBottom: '0.5rem', background: `${statusColor}18`, color: statusColor, border: `1px solid ${statusColor}33` }}>
                    {isFailed ? 'Failed' : isComplete ? 'Complete' : isRunning ? 'Measuring' : 'Waiting for doctor'}
                  </div>
                  <h2 style={{ fontSize: '1rem', fontWeight: 700, color: C.navy, marginBottom: '0.25rem', lineHeight: 1.3 }}>
                    {isComplete ? 'Measurement complete'
                      : isFailed ? 'Measurement could not complete'
                      : isRunning ? (isPalmLive ? 'Show your palm to the camera' : 'Look directly at the camera')
                      : 'Your doctor will start the measurement'}
                  </h2>
                  <p style={{ fontSize: 13, color: C.bodyText, lineHeight: 1.6 }}>
                    {isComplete ? 'Your doctor has received your results. Thank you.'
                      : isFailed ? 'Please improve your lighting and ask your doctor to start again.'
                      : isRunning ? (isPalmLive
                          ? 'Hold the inside of your hand flat and open, centered about 30 cm from the camera.'
                          : 'Stay still while we detect your face and calibrate to your signal.')
                      : 'Once started, follow the on-screen instructions.'}
                  </p>
                  {isIdle && submitted && (
                    <div style={{ marginTop: 10, display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: C.blue, fontWeight: 500, padding: '4px 12px', borderRadius: 20, background: C.blueLight, border: `1px solid ${C.blue}33` }}>
                      Mode: {selectedMode === 'palm' ? 'Palm' : 'Auto'}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Card body */}
            <div style={{ padding: '1.5rem' }}>

              {/* Steps progress */}
              {(isRunning || isComplete) && (
                <div style={{ marginBottom: isComplete ? '1.5rem' : 0 }}>
                  <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.muted, marginBottom: '0.75rem' }}>Measurement progress</p>
                  {STEPS.map(({ id, label }, i) => {
                    const done   = doneSteps.includes(id)
                    const prevId = STEPS[i - 1]?.id
                    const active = !done && isRunning && (i === 0 || doneSteps.includes(prevId))
                    return <StepRow key={id} label={label} done={done} active={active} last={i === STEPS.length - 1} />
                  })}
                </div>
              )}

              {/* Results */}
              {isComplete && final.hr_bpm && (
                <div>
                  <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: C.muted, marginBottom: '0.75rem', paddingTop: '1rem', borderTop: `1px solid ${C.borderLt}` }}>Your results</p>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
                    <ResultCard label="Heart rate" value={final.hr_bpm} unit="BPM" accent delay={0} />
                    {final.rr_bpm && <ResultCard label="Respiratory" value={final.rr_bpm} unit="BrPM" delay={150} />}
                  </div>
                  {final.route_palm && (
                    <div style={{ marginTop: 10, padding: '10px 16px', borderRadius: 12, background: C.greenBg, border: `1px solid ${C.green}33`, fontSize: 12, color: C.green, fontWeight: 500, textAlign: 'center' }}>
                      Measured via palm signal for improved accuracy
                    </div>
                  )}
                  <div style={{ marginTop: 16, padding: '14px 16px', borderRadius: 12, background: C.blueLight, border: `1px solid ${C.blue}22`, fontSize: 13, color: C.blue, lineHeight: 1.5 }}>
                    Your doctor has received your results and will discuss them with you shortly.
                  </div>
                </div>
              )}

              {/* Failed */}
              {isFailed && (
                <div style={{ padding: '14px 16px', borderRadius: 12, background: C.roseBg, border: `1px solid ${C.rose}44`, fontSize: 13, color: C.rose, lineHeight: 1.6 }}>
                  <strong style={{ fontWeight: 700 }}>What to do next:</strong>
                  <ul style={{ marginTop: 8, paddingLeft: 18, lineHeight: 1.8 }}>
                    <li>Move closer to a window or lamp</li>
                    <li>Ensure your face is well lit from the front</li>
                    <li>Ask your doctor to start a new measurement</li>
                  </ul>
                </div>
              )}
            </div>
          </div>

          <p style={{ textAlign: 'center', marginTop: '1.25rem', fontSize: 11, color: C.muted }}>
            PulseRoute — Investigational use only — rPPG technology
          </p>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.85); } }
        @keyframes fadeInOut { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${C.bg}; }
      `}</style>
    </div>
  )
}