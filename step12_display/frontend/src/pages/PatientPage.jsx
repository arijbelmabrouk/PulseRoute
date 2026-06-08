import { useWebSocket } from '../hooks/useWebSocket'
import { useEffect, useState } from 'react'

const C = {
  blue:      '#1a5dab',
  blueLight: '#e8f1fb',
  blueMid:   '#2d79d6',
  navy:      '#1a2b4a',
  bodyText:  '#3d4f6b',
  muted:     '#6b7a91',
  border:    '#dce4ef',
  borderLt:  '#edf2f8',
  bg:        '#f5f7fa',
  white:     '#ffffff',
  green:     '#1a9e6e',
  greenBg:   '#eaf7f1',
  amber:     '#c47a10',
  amberBg:   '#fdf3e1',
  rose:      '#c0392b',
  roseBg:    '#fcecea',
  sans:      "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
}

const STEPS_FACE = [
  { id: 1,  label: 'Camera initialised' },
  { id: 2,  label: 'Face detected and calibrated' },
  { id: 3,  label: 'Signal captured' },
  { id: 5,  label: 'Pulse waveform extracted' },
  { id: 6,  label: 'Signal filtered' },
  { id: 7,  label: 'Frequency analysis' },
  { id: 9,  label: 'Heart rate computed' },
  { id: 10, label: 'Respiratory rate computed' },
  { id: 11, label: 'Quality assessment' },
]
const STEPS_PALM = [
  { id: 2,  label: 'Palm detected and calibrated' },
  { id: 3,  label: 'Palm signal captured' },
  { id: 5,  label: 'Pulse waveform extracted' },
  { id: 6,  label: 'Signal filtered' },
  { id: 7,  label: 'Frequency analysis' },
  { id: 9,  label: 'Heart rate computed' },
  { id: 10, label: 'Respiratory rate computed' },
  { id: 11, label: 'Quality assessment' },
]

function ArcRing({ pct, failed, complete }) {
  const size = 160, cx = 80, cy = 80, r = 66, sw = 5
  const toRad = d => (d * Math.PI) / 180
  const polar = (a, rd) => ({ x: cx + rd * Math.cos(toRad(a)), y: cy + rd * Math.sin(toRad(a)) })
  const arcPath = (sa, ea, rd) => {
    const s = polar(sa, rd), e = polar(ea, rd)
    const large = (ea - sa) > 180 ? 1 : 0
    return `M ${s.x} ${s.y} A ${rd} ${rd} 0 ${large} 1 ${e.x} ${e.y}`
  }
  const startDeg = -220, totalArc = 280
  const endDeg   = startDeg + (pct / 100) * totalArc
  const trackEnd = startDeg + totalArc
  const fillColor = failed ? C.rose : complete ? C.green : C.blue

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <path d={arcPath(startDeg, trackEnd, r)} fill="none"
        stroke={C.borderLt} strokeWidth={sw} strokeLinecap="round" />
      <path d={arcPath(startDeg, Math.max(startDeg + 0.5, endDeg), r)}
        fill="none" stroke={fillColor} strokeWidth={sw} strokeLinecap="round"
        style={{ transition: 'all 0.6s ease' }} />
      <text x={cx} y={cy - 6} textAnchor="middle"
        fontSize="26" fontWeight="700" fill={failed ? C.rose : complete ? C.green : C.navy}
        fontFamily="Inter, sans-serif" letterSpacing="-1">
        {failed ? '—' : `${pct}`}
      </text>
      <text x={cx} y={cy + 14} textAnchor="middle"
        fontSize="10" fill={C.muted}
        fontFamily="Inter, sans-serif" letterSpacing="1">
        {failed ? 'FAILED' : complete ? 'DONE' : 'PROGRESS'}
      </text>
    </svg>
  )
}

function StepRow({ label, done, active }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '9px 0',
      borderBottom: `1px solid ${C.borderLt}`,
      opacity: done || active ? 1 : 0.4,
      transition: 'opacity 0.35s',
    }}>
      <div style={{
        width: 22, height: 22, borderRadius: 6, flexShrink: 0,
        background: done ? C.blue : active ? C.blueLight : C.bg,
        border: `1.5px solid ${done ? C.blue : active ? C.blue : C.border}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        transition: 'all 0.3s',
      }}>
        {done && (
          <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
            <path d="M1 4l3 3 5-6" stroke={C.white} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
        {active && !done && (
          <div style={{
            width: 7, height: 7, borderRadius: '50%', background: C.blue,
            animation: 'blink 1.2s infinite',
          }} />
        )}
      </div>
      <span style={{
        fontSize: 13, fontFamily: C.sans,
        color: done ? C.navy : active ? C.blue : C.muted,
        fontWeight: done ? 500 : 400,
        transition: 'color 0.3s',
      }}>{label}</span>
      {done && (
        <span style={{
          marginLeft: 'auto', fontSize: 10, fontWeight: 600,
          color: C.green, letterSpacing: '0.05em', textTransform: 'uppercase',
        }}>Done</span>
      )}
    </div>
  )
}

function ResultCard({ label, value, unit, accent }) {
  const [shown, setShown] = useState(false)
  useEffect(() => { if (value) setTimeout(() => setShown(true), 100) }, [!!value])

  return (
    <div style={{
      background: accent ? C.blueLight : C.bg,
      border: `1px solid ${accent ? C.blue + '44' : C.border}`,
      borderRadius: 12, padding: '1.25rem',
      textAlign: 'center',
      opacity: shown ? 1 : 0,
      transform: shown ? 'translateY(0)' : 'translateY(8px)',
      transition: 'opacity 0.45s ease, transform 0.45s ease',
    }}>
      <div style={{
        fontSize: '2.2rem', fontWeight: 700,
        color: accent ? C.blue : C.navy, lineHeight: 1,
        letterSpacing: '-0.03em', fontFamily: C.sans,
      }}>{value}</div>
      <div style={{
        fontSize: 11, fontWeight: 600, letterSpacing: '0.07em',
        textTransform: 'uppercase', color: C.muted, marginTop: 6,
        fontFamily: C.sans,
      }}>{label} · {unit}</div>
    </div>
  )
}

export default function PatientPage() {
  const { state, connected } = useWebSocket('/ws/patient')

  const isFailed   = state.status === 'failed'
  const isIdle     = state.status === 'idle'
  const isRunning  = state.status === 'running' || state.status === 'starting'
  const isComplete = state.status === 'complete'
  const isPalm     = state.modality === 'palm' || state.mode === 'palm'
  const isRouting  = isRunning && state.route_palm === true

  const final     = state.final || {}
  const pct       = state.progress || 0
  const doneSteps = Object.keys(state.steps || {}).map(Number)
  const STEPS     = isPalm ? STEPS_PALM : STEPS_FACE

  const instruction = isPalm
    ? 'Hold your palm flat and open, centered in front of the camera'
    : 'Look directly at the camera and remain still'

  const statusLabel = isFailed ? 'Measurement failed'
    : isComplete ? 'Measurement complete'
    : isRunning  ? 'Measuring your vitals'
    : 'Waiting for your doctor to start'

  const statusColor = isFailed ? C.rose
    : isComplete ? C.green
    : isRunning  ? C.blue
    : C.muted

  return (
    <div style={{
      minHeight: '100vh', background: C.bg,
      fontFamily: C.sans, color: C.navy,
      display: 'flex', flexDirection: 'column',
    }}>

      {/* ── Nav ── */}
      <nav style={{
        background: C.white, borderBottom: `1px solid ${C.border}`,
        padding: '0 2rem', height: 56,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 30, height: 30, borderRadius: '50%',
            border: `2.5px solid ${C.blue}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{ width: 14, height: 14, borderRadius: '50%', border: `2px solid ${C.blue}` }} />
          </div>
          <span style={{ fontWeight: 700, fontSize: 15, color: C.navy }}>PulseRoute</span>
        </div>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          fontSize: 12, color: connected ? C.green : C.muted, fontWeight: 500,
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: connected ? C.green : C.muted,
            animation: connected ? 'blink 2s infinite' : 'none',
          }} />
          {connected ? 'Connected' : 'Connecting…'}
        </div>
      </nav>

      {/* Progress bar */}
      <div style={{ height: 3, background: C.borderLt }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: isFailed ? C.rose : isComplete ? C.green : C.blue,
          transition: 'width 0.5s ease', borderRadius: '0 3px 3px 0',
        }} />
      </div>

      {/* ── Content ── */}
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center',
        justifyContent: 'center', padding: '2rem',
      }}>
        <div style={{ width: '100%', maxWidth: 480 }}>

          {/* Header */}
          <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
            <h1 style={{
              fontSize: '1.4rem', fontWeight: 700, color: C.navy,
              marginBottom: '0.3rem', letterSpacing: '-0.02em',
            }}>Contactless Vital Measurement</h1>
            <p style={{ fontSize: 13, color: C.muted }}>
              PulseRoute · Remote photoplethysmography
            </p>
          </div>

          {/* Main card */}
          <div style={{
            background: C.white,
            border: `1px solid ${isFailed ? C.rose + '66' : C.border}`,
            borderRadius: 16,
            overflow: 'hidden',
            transition: 'border-color 0.3s',
          }}>

            <div style={{ padding: '1.5rem' }}>

              {/* Ring + status */}
              <div style={{
                display: 'flex', alignItems: 'center',
                gap: '1.5rem', marginBottom: '1.25rem',
              }}>
                <div style={{ flexShrink: 0 }}>
                  <ArcRing pct={pct} failed={isFailed} complete={isComplete} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: 7,
                    fontSize: 12, fontWeight: 600,
                    padding: '5px 14px', borderRadius: 20, marginBottom: '0.6rem',
                    background: isFailed ? C.roseBg : isComplete ? C.greenBg : isRunning ? C.blueLight : C.bg,
                    color: statusColor,
                    border: `1px solid ${statusColor}33`,
                  }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: '50%', background: statusColor,
                      animation: isRunning ? 'blink 1.4s infinite' : 'none',
                    }} />
                    {statusLabel}
                  </div>

                  <p style={{
                    fontSize: 13, color: C.muted, lineHeight: 1.6,
                  }}>
                    {isFailed
                      ? (state.message || 'The signal was too weak for a reliable result.')
                      : isRouting
                        ? 'Switching to palm measurement for improved accuracy.'
                        : isRunning
                          ? instruction
                          : isComplete
                            ? 'Your doctor has received your results.'
                            : 'Your doctor will start the measurement from their interface.'}
                  </p>
                </div>
              </div>

              {/* Palm instruction */}
              {isRouting && (
                <div style={{
                  padding: '12px 14px', borderRadius: 10, marginBottom: '1rem',
                  background: C.amberBg, border: `1px solid ${C.amber}44`,
                  fontSize: 13, color: C.amber, lineHeight: 1.6,
                }}>
                  Please hold the inside of your palm flat to the camera — open, centered, approximately 30 cm from the lens. Keep it still.
                </div>
              )}

              {/* Motion warning */}
              {isRunning && state.motion_pct > 20 && (
                <div style={{
                  padding: '10px 14px', borderRadius: 10, marginBottom: '1rem',
                  background: C.amberBg, border: `1px solid ${C.amber}44`,
                  fontSize: 13, color: C.amber,
                }}>
                  Movement detected — please hold still
                </div>
              )}

              {/* Failed advice */}
              {isFailed && (
                <div style={{
                  padding: '12px 14px', borderRadius: 10,
                  background: C.roseBg, border: `1px solid ${C.rose}44`,
                  fontSize: 13, color: C.rose, lineHeight: 1.6,
                }}>
                  Move closer to a light source and ensure your face is well-lit. Your doctor will start a new measurement.
                </div>
              )}

              {/* Steps list */}
              {(isRunning || isComplete) && (
                <div style={{
                  borderTop: `1px solid ${C.borderLt}`,
                  paddingTop: '1rem', marginTop: '0.5rem',
                }}>
                  <p style={{
                    fontSize: 11, fontWeight: 700, letterSpacing: '0.07em',
                    textTransform: 'uppercase', color: C.muted, marginBottom: '0.75rem',
                  }}>Measurement progress</p>
                  {STEPS.map(({ id, label }, i) => {
                    const done   = doneSteps.includes(id)
                    const active = !done && isRunning && doneSteps.includes(STEPS[i - 1]?.id)
                    return <StepRow key={id} label={label} done={done} active={active} />
                  })}
                </div>
              )}

              {/* Results */}
              {isComplete && final.hr_bpm && (
                <div style={{
                  borderTop: `1px solid ${C.borderLt}`,
                  paddingTop: '1rem', marginTop: '1rem',
                }}>
                  <p style={{
                    fontSize: 11, fontWeight: 700, letterSpacing: '0.07em',
                    textTransform: 'uppercase', color: C.muted, marginBottom: '0.75rem',
                  }}>Your results</p>
                  <div style={{
                    display: 'grid', gridTemplateColumns: '1fr 1fr',
                    gap: 10,
                  }}>
                    <ResultCard label="Heart rate" value={final.hr_bpm} unit="BPM" accent />
                    {final.rr_bpm && <ResultCard label="Respiratory" value={final.rr_bpm} unit="BrPM" />}
                  </div>
                  {final.route_palm && (
                    <div style={{
                      marginTop: 10, padding: '9px 14px', borderRadius: 20,
                      background: C.greenBg, border: `1px solid ${C.green}44`,
                      fontSize: 12, color: C.green, fontWeight: 500,
                      textAlign: 'center',
                    }}>
                      Measured via palm signal for improved accuracy
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          <p style={{
            textAlign: 'center', marginTop: '1.25rem',
            fontSize: 11, color: C.muted,
          }}>
            PulseRoute · Investigational use only
          </p>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
        * { box-sizing: border-box; margin: 0; padding: 0; }
      `}</style>
    </div>
  )
}