import { useWebSocket } from '../hooks/useWebSocket'
import { useEffect, useState, useRef } from 'react'

// ── Design tokens — Circle Health inspired ──────────
const C = {
  blue:      '#0072CE',   // Circle Health primary blue
  blueDark:  '#005BA4',
  blueLight: '#E8F4FF',
  blueMid:   '#3B90D8',
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

// ── Step definitions ─────────────────────────────────
const STEPS_FACE = [
  { id: 1,  label: 'Camera initialised',              icon: '📷' },
  { id: 2,  label: 'Face detected and calibrated',    icon: '👤' },
  { id: 3,  label: 'Signal captured',                 icon: '📶' },
  { id: 5,  label: 'Pulse waveform extracted',        icon: '〰️' },
  { id: 6,  label: 'Signal filtered',                 icon: '🔍' },
  { id: 9,  label: 'Heart rate computed',             icon: '❤️' },
  { id: 10, label: 'Respiratory rate computed',       icon: '🫁' },
  { id: 11, label: 'Quality assessment complete',     icon: '✅' },
]

const STEPS_PALM = [
  { id: 2,  label: 'Palm detected and calibrated',    icon: '✋' },
  { id: 3,  label: 'Palm signal captured',            icon: '📶' },
  { id: 5,  label: 'Pulse waveform extracted',        icon: '〰️' },
  { id: 6,  label: 'Signal filtered',                 icon: '🔍' },
  { id: 9,  label: 'Heart rate computed',             icon: '❤️' },
  { id: 10, label: 'Respiratory rate computed',       icon: '🫁' },
  { id: 11, label: 'Quality assessment complete',     icon: '✅' },
]

// ── Helpers ──────────────────────────────────────────
function getInstruction(state) {
  const { status, modality, route_palm, progress } = state

  if (status === 'idle' || status === undefined) {
    return {
      title: 'Your doctor will start the measurement',
      body:  'Please wait. Once started, follow the on-screen instructions.',
      icon:  '⏳',
    }
  }
  if (status === 'starting') {
    return {
      title: 'Starting measurement…',
      body:  'Please ensure your camera is working and your face is visible.',
      icon:  '📷',
    }
  }
  if (status === 'failed') {
    return {
      title: 'Measurement could not complete',
      body:  'Please improve your lighting and ask your doctor to start again.',
      icon:  '⚠️',
    }
  }
  if (status === 'complete') {
    return {
      title: 'Measurement complete',
      body:  'Your doctor has received your results. Thank you.',
      icon:  '✓',
    }
  }

  // Running — phase-specific instructions
  if (modality === 'palm' || route_palm) {
    return {
      title: 'Show your palm to the camera',
      body:  'Hold the inside of your hand flat, open, and centered — approximately 30 cm from the camera. Keep it still.',
      icon:  '✋',
      isPalm: true,
    }
  }

  if (progress < 20) {
    return {
      title: 'Look directly at the camera',
      body:  'Stay still while we detect your face and calibrate to your personal signal.',
      icon:  '👤',
    }
  }

  return {
    title: 'Look directly at the camera',
    body:  'Remain still and breathe normally. The measurement takes about 35 seconds.',
    icon:  '👁️',
  }
}

// ── Components ───────────────────────────────────────

function StepRow({ label, icon, done, active, last }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      padding: '10px 0',
      borderBottom: last ? 'none' : `1px solid ${C.borderLt}`,
      opacity: done || active ? 1 : 0.35,
      transition: 'opacity 0.4s ease',
    }}>
      {/* Status indicator */}
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
            background: C.blue,
            animation: 'pulse 1.4s ease infinite',
          }} />
        ) : (
          <span style={{ fontSize: 11 }}>{icon}</span>
        )}
      </div>

      <span style={{
        fontSize: 13, fontFamily: C.sans,
        color: done ? C.navy : active ? C.blue : C.muted,
        fontWeight: done ? 600 : active ? 500 : 400,
        transition: 'color 0.3s',
        flex: 1,
      }}>{label}</span>

      {done && (
        <span style={{
          fontSize: 10, fontWeight: 700,
          color: C.green, letterSpacing: '0.07em',
          textTransform: 'uppercase',
        }}>Done</span>
      )}
      {active && !done && (
        <span style={{
          fontSize: 10, fontWeight: 700,
          color: C.blue, letterSpacing: '0.07em',
          textTransform: 'uppercase',
          animation: 'fadeInOut 1.5s ease infinite',
        }}>Active</span>
      )}
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
      borderRadius: 16,
      padding: '1.5rem 1rem',
      textAlign: 'center',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0) scale(1)' : 'translateY(12px) scale(0.97)',
      transition: 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)',
      boxShadow: accent ? `0 8px 24px ${C.blue}33` : '0 2px 8px rgba(0,0,0,0.06)',
    }}>
      <div style={{
        fontSize: '2.4rem', fontWeight: 700, lineHeight: 1,
        color: accent ? C.white : C.navy,
        letterSpacing: '-0.03em', fontFamily: C.sans,
        marginBottom: 6,
      }}>{value ?? '—'}</div>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
        textTransform: 'uppercase',
        color: accent ? 'rgba(255,255,255,0.75)' : C.muted,
        fontFamily: C.sans,
      }}>{label}</div>
      {unit && (
        <div style={{
          fontSize: 11,
          color: accent ? 'rgba(255,255,255,0.6)' : C.muted,
          marginTop: 2, fontFamily: C.sans,
        }}>{unit}</div>
      )}
    </div>
  )
}

// ── Main PatientPage ─────────────────────────────────

export default function PatientPage() {
  const { state, connected, latestFrame } = useWebSocket('/ws/patient')

  const isFailed   = state.status === 'failed'
  const isIdle     = !state.status || state.status === 'idle'
  const isRunning  = state.status === 'running' || state.status === 'starting'
  const isComplete = state.status === 'complete'
  const isPalm     = state.modality === 'palm' ||
                     state.mode === 'palm' ||
                     state.route_palm === true

  const final      = state.final || {}
  const steps      = state.steps || {}
  const doneSteps  = Object.keys(steps).map(Number)
  const STEPS      = isPalm ? STEPS_PALM : STEPS_FACE
  const pct        = state.progress || 0
  const motionHigh = (state.motion_pct || 0) > 20

  const instruction = getInstruction({
    status:    state.status,
    modality:  state.modality,
    route_palm: state.route_palm,
    progress:  pct,
  })

  // Show frame feed only when running and a frame exists
  const showFeed = isRunning && latestFrame != null

  // Status color
  const statusColor = isFailed ? C.rose
    : isComplete ? C.green
    : isRunning  ? C.blue
    : C.muted

  return (
    <div style={{
      minHeight: '100vh',
      background: C.bg,
      fontFamily: C.sans,
      color: C.navy,
      display: 'flex',
      flexDirection: 'column',
    }}>

      {/* ── Nav ── */}
      <nav style={{
        background: C.white,
        borderBottom: `1px solid ${C.border}`,
        padding: '0 2rem',
        height: 60,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky', top: 0, zIndex: 20,
        boxShadow: '0 1px 4px rgba(0,114,206,0.06)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Logo mark */}
          <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
            <circle cx="17" cy="17" r="16"
              stroke={C.blue} strokeWidth="2.5" fill="none"/>
            <circle cx="17" cy="17" r="8"
              stroke={C.blue} strokeWidth="2" fill="none"/>
            <circle cx="17" cy="17" r="3"
              fill={C.blue}/>
          </svg>
          <div>
            <span style={{
              fontWeight: 700, fontSize: 16, color: C.navy,
              letterSpacing: '-0.01em',
            }}>PulseRoute</span>
            <span style={{
              display: 'block', fontSize: 10, color: C.muted,
              letterSpacing: '0.05em', textTransform: 'uppercase',
            }}>Patient Monitoring</span>
          </div>
        </div>

        {/* Connection status */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 7,
          fontSize: 12, fontWeight: 500,
          color: connected ? C.green : C.muted,
          background: connected ? C.greenBg : C.bg,
          padding: '5px 14px', borderRadius: 20,
          border: `1px solid ${connected ? C.green + '44' : C.border}`,
          transition: 'all 0.3s',
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: connected ? C.green : C.muted,
            animation: connected ? 'pulse 2s ease infinite' : 'none',
          }} />
          {connected ? 'Connected to clinic' : 'Reconnecting…'}
        </div>
      </nav>

      {/* Progress bar */}
      <div style={{ height: 3, background: C.borderLt, flexShrink: 0 }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: isFailed ? C.rose : isComplete ? C.green : C.blue,
          transition: 'width 0.6s ease',
          borderRadius: '0 3px 3px 0',
          boxShadow: isRunning ? `0 0 8px ${C.blue}66` : 'none',
        }} />
      </div>

      {/* ── Main content ── */}
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        padding: '2rem 1rem',
      }}>
        <div style={{ width: '100%', maxWidth: 520 }}>

          {/* Page header */}
          <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
            <h1 style={{
              fontSize: '1.5rem', fontWeight: 700, color: C.navy,
              marginBottom: '0.4rem', letterSpacing: '-0.025em',
              fontFamily: C.sans,
            }}>Contactless Vital Signs</h1>
            <p style={{
              fontSize: 13, color: C.muted, fontFamily: C.sans,
            }}>
              Remote measurement via camera · No contact required
            </p>
          </div>

          {/* ── Main card ── */}
          <div style={{
            background: C.white,
            border: `1.5px solid ${isFailed ? C.rose + '55' : C.border}`,
            borderRadius: 20,
            overflow: 'hidden',
            boxShadow: '0 4px 20px rgba(0,114,206,0.08)',
            transition: 'border-color 0.3s',
          }}>

            {/* ── Camera feed OR status header ── */}
            {showFeed ? (
              /* Live camera feed — replaces arc ring during measurement */
              <div style={{ position: 'relative', background: '#0a0f1a' }}>
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
                {/* Live badge */}
                <div style={{
                  position: 'absolute', top: 14, left: 14,
                  display: 'flex', alignItems: 'center', gap: 6,
                  background: 'rgba(0,0,0,0.65)',
                  backdropFilter: 'blur(4px)',
                  padding: '5px 12px', borderRadius: 20,
                  fontSize: 11, fontWeight: 700,
                  color: C.white, letterSpacing: '0.07em',
                  textTransform: 'uppercase',
                }}>
                  <span style={{
                    width: 6, height: 6, borderRadius: '50%',
                    background: '#FF3B30',
                    animation: 'pulse 1.2s ease infinite',
                  }} />
                  Live
                </div>
                {/* Modality badge */}
                <div style={{
                  position: 'absolute', top: 14, right: 14,
                  background: isPalm
                    ? 'rgba(224,123,0,0.85)'
                    : 'rgba(0,114,206,0.85)',
                  backdropFilter: 'blur(4px)',
                  padding: '4px 12px', borderRadius: 20,
                  fontSize: 11, fontWeight: 700,
                  color: C.white, letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                }}>
                  {isPalm ? 'Palm' : 'Face'}
                </div>
                {/* Instruction overlay at bottom */}
                <div style={{
                  position: 'absolute', bottom: 0, left: 0, right: 0,
                  background: 'linear-gradient(transparent, rgba(0,0,0,0.72))',
                  padding: '2rem 1.25rem 1rem',
                }}>
                  <p style={{
                    color: C.white, fontSize: 14, fontWeight: 600,
                    fontFamily: C.sans, lineHeight: 1.4,
                    textShadow: '0 1px 3px rgba(0,0,0,0.4)',
                  }}>{instruction.title}</p>
                  <p style={{
                    color: 'rgba(255,255,255,0.78)', fontSize: 12,
                    fontFamily: C.sans, marginTop: 3, lineHeight: 1.5,
                  }}>{instruction.body}</p>
                </div>
                {/* Motion warning bar */}
                {motionHigh && (
                  <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0,
                    background: C.amber,
                    padding: '8px 16px',
                    fontSize: 12, fontWeight: 600, color: C.white,
                    textAlign: 'center', fontFamily: C.sans,
                    letterSpacing: '0.02em',
                  }}>
                    ⚠ Movement detected — please hold still
                  </div>
                )}
              </div>
            ) : (
              /* Status header — shown when idle / complete / failed / no frame yet */
              <div style={{
                padding: '2rem',
                background: isFailed ? C.roseBg
                  : isComplete ? C.greenBg
                  : isRunning  ? C.blueLight
                  : C.bg,
                borderBottom: `1px solid ${C.border}`,
                display: 'flex', alignItems: 'center', gap: '1.5rem',
                transition: 'background 0.4s',
              }}>
                {/* Status icon */}
                <div style={{
                  width: 64, height: 64, borderRadius: '50%', flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: isFailed ? C.rose + '22'
                    : isComplete ? C.green + '22'
                    : isRunning  ? C.blue + '22'
                    : C.border + '44',
                  border: `2px solid ${statusColor}44`,
                }}>
                  <span style={{ fontSize: '1.75rem' }}>
                    {isFailed ? '⚠️'
                      : isComplete ? '✓'
                      : isRunning  ? instruction.icon
                      : '⏳'}
                  </span>
                </div>

                <div style={{ flex: 1 }}>
                  {/* Status badge */}
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: 6,
                    fontSize: 11, fontWeight: 700,
                    letterSpacing: '0.07em', textTransform: 'uppercase',
                    padding: '4px 12px', borderRadius: 20,
                    marginBottom: '0.5rem',
                    background: `${statusColor}18`,
                    color: statusColor,
                    border: `1px solid ${statusColor}33`,
                  }}>
                    <span style={{
                      width: 5, height: 5, borderRadius: '50%',
                      background: statusColor,
                      animation: isRunning ? 'pulse 1.4s ease infinite' : 'none',
                    }} />
                    {isFailed ? 'Failed'
                      : isComplete ? 'Complete'
                      : isRunning  ? 'Measuring'
                      : 'Waiting'}
                  </div>

                  <h2 style={{
                    fontSize: '1rem', fontWeight: 700, color: C.navy,
                    marginBottom: '0.25rem', fontFamily: C.sans,
                    lineHeight: 1.3,
                  }}>{instruction.title}</h2>
                  <p style={{
                    fontSize: 13, color: C.bodyText, lineHeight: 1.6,
                    fontFamily: C.sans,
                  }}>{instruction.body}</p>
                </div>
              </div>
            )}

            {/* ── Card body ── */}
            <div style={{ padding: '1.5rem' }}>

              {/* Palm instruction banner */}
              {isRunning && instruction.isPalm && (
                <div style={{
                  padding: '14px 16px', borderRadius: 12,
                  marginBottom: '1.25rem',
                  background: C.amberBg,
                  border: `1px solid ${C.amber}44`,
                  fontSize: 13, color: C.amber,
                  lineHeight: 1.6, fontFamily: C.sans,
                  display: 'flex', gap: 10, alignItems: 'flex-start',
                }}>
                  <span style={{ fontSize: '1.2rem', flexShrink: 0 }}>✋</span>
                  <div>
                    <strong style={{ fontWeight: 700 }}>Show your palm to the camera.</strong>
                    {' '}Hold the inside of your hand flat and open, centered in the frame, about 30 cm from the lens. Keep it still.
                  </div>
                </div>
              )}

              {/* Measurement steps */}
              {(isRunning || isComplete) && (
                <div style={{ marginBottom: isComplete ? '1.5rem' : 0 }}>
                  <p style={{
                    fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
                    textTransform: 'uppercase', color: C.muted,
                    marginBottom: '0.75rem', fontFamily: C.sans,
                  }}>Measurement progress</p>
                  {STEPS.map(({ id, label, icon }, i) => {
                    const done   = doneSteps.includes(id)
                    const prevId = STEPS[i - 1]?.id
                    const active = !done && isRunning &&
                                   (i === 0 || doneSteps.includes(prevId))
                    return (
                      <StepRow
                        key={id}
                        label={label}
                        icon={icon}
                        done={done}
                        active={active}
                        last={i === STEPS.length - 1}
                      />
                    )
                  })}
                </div>
              )}

              {/* Results */}
              {isComplete && final.hr_bpm && (
                <div>
                  <p style={{
                    fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
                    textTransform: 'uppercase', color: C.muted,
                    marginBottom: '0.75rem', fontFamily: C.sans,
                    paddingTop: '1rem',
                    borderTop: `1px solid ${C.borderLt}`,
                  }}>Your results</p>
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: 10,
                  }}>
                    <ResultCard
                      label="Heart rate"
                      value={final.hr_bpm}
                      unit="BPM"
                      accent
                      delay={0}
                    />
                    {final.rr_bpm && (
                      <ResultCard
                        label="Respiratory"
                        value={final.rr_bpm}
                        unit="BrPM"
                        delay={150}
                      />
                    )}
                  </div>

                  {/* Routing note */}
                  {final.route_palm && (
                    <div style={{
                      marginTop: 10, padding: '10px 16px', borderRadius: 12,
                      background: C.greenBg,
                      border: `1px solid ${C.green}33`,
                      fontSize: 12, color: C.green, fontWeight: 500,
                      textAlign: 'center', fontFamily: C.sans,
                    }}>
                      Measured via palm signal for improved accuracy
                    </div>
                  )}

                  {/* Completion message */}
                  <div style={{
                    marginTop: 16, padding: '14px 16px', borderRadius: 12,
                    background: C.blueLight,
                    border: `1px solid ${C.blue}22`,
                    fontSize: 13, color: C.blue,
                    fontFamily: C.sans, lineHeight: 1.5,
                    display: 'flex', gap: 10, alignItems: 'center',
                  }}>
                    <span style={{ fontSize: '1.25rem' }}>👨‍⚕️</span>
                    Your doctor has received your results and will discuss them with you shortly.
                  </div>
                </div>
              )}

              {/* Failed state */}
              {isFailed && (
                <div style={{
                  padding: '14px 16px', borderRadius: 12,
                  background: C.roseBg,
                  border: `1px solid ${C.rose}44`,
                  fontSize: 13, color: C.rose,
                  fontFamily: C.sans, lineHeight: 1.6,
                }}>
                  <strong style={{ fontWeight: 700 }}>What to do next:</strong>
                  <ul style={{
                    marginTop: 8, paddingLeft: 18, lineHeight: 1.8,
                  }}>
                    <li>Move closer to a window or lamp</li>
                    <li>Ensure your face is well lit</li>
                    <li>Ask your doctor to start a new measurement</li>
                  </ul>
                </div>
              )}

              {/* Idle — waiting message */}
              {isIdle && (
                <div style={{
                  textAlign: 'center', padding: '1.5rem 1rem',
                  color: C.muted, fontSize: 13, fontFamily: C.sans,
                  lineHeight: 1.7,
                }}>
                  <div style={{
                    width: 48, height: 48, borderRadius: '50%',
                    background: C.blueLight,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    margin: '0 auto 1rem',
                    fontSize: '1.5rem',
                  }}>🏥</div>
                  <p style={{ fontWeight: 600, color: C.navy, marginBottom: 6 }}>
                    Ready when your doctor is
                  </p>
                  <p style={{ fontSize: 12 }}>
                    Ensure your camera is on and your face is clearly visible in good lighting.
                    Your doctor will start the measurement from their dashboard.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Footer */}
          <p style={{
            textAlign: 'center', marginTop: '1.25rem',
            fontSize: 11, color: C.muted, fontFamily: C.sans,
          }}>
            PulseRoute · Investigational use only · rPPG technology
          </p>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.5; transform: scale(0.85); }
        }
        @keyframes fadeInOut {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.4; }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${C.bg}; }
      `}</style>
    </div>
  )
}