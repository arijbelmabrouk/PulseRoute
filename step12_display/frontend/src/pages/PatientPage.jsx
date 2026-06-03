import { useWebSocket } from '../hooks/useWebSocket'

const STEPS = [
  { id: 1,  label: 'Camera ready' },
  { id: 2,  label: 'Face detected & calibrated' },
  { id: 3,  label: 'Signal captured' },
  { id: 5,  label: 'Pulse extracted' },
  { id: 6,  label: 'Signal filtered' },
  { id: 7,  label: 'Frequency analysis' },
  { id: 9,  label: 'Heart rate computed' },
  { id: 10, label: 'Respiratory rate computed' },
  { id: 11, label: 'Quality assessment' },
]

function ProgressRing({ pct }) {
  const r = 54, circ = 2 * Math.PI * r
  const offset = circ - (pct / 100) * circ
  return (
    <svg width="128" height="128" viewBox="0 0 128 128">
      <circle cx="64" cy="64" r={r}
        fill="none" stroke="#e5e7eb" strokeWidth="6" />
      <circle cx="64" cy="64" r={r}
        fill="none" stroke="#1a56db" strokeWidth="6"
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round"
        transform="rotate(-90 64 64)"
        style={{ transition: 'stroke-dashoffset 0.6s ease' }}
      />
      <text x="64" y="60" textAnchor="middle"
        fontSize="22" fontWeight="600" fill="#0f1923"
        fontFamily="Inter, sans-serif">
        {pct}%
      </text>
      <text x="64" y="76" textAnchor="middle"
        fontSize="11" fill="#8a95a3"
        fontFamily="Inter, sans-serif">
        complete
      </text>
    </svg>
  )
}

export default function PatientPage() {
  const { state, connected } = useWebSocket('/ws/patient')
  const isIdle     = state.status === 'idle'
  const isRunning  = state.status === 'running' || state.status === 'starting'
  const isComplete = state.status === 'complete'
  const final      = state.final || {}
  const pct        = state.progress || 0
  const doneSteps  = Object.keys(state.steps || {}).map(Number)

  return (
    <div style={{
      minHeight: '100vh', background: 'var(--bg)',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      padding: '2rem',
    }}>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '8px',
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: '999px', padding: '5px 14px',
          fontSize: '12px', color: 'var(--text2)',
          marginBottom: '1rem', boxShadow: 'var(--shadow-sm)',
        }}>
          <span className={connected ? 'blink' : ''} style={{
            width: 7, height: 7, borderRadius: '50%',
            background: connected ? 'var(--success)' : '#d1d5db',
            display: 'inline-block', flexShrink: 0,
          }} />
          {connected ? 'Connected to measurement system' : 'Connecting…'}
        </div>
        <h1 style={{
          fontSize: '1.75rem', fontWeight: 600,
          color: 'var(--text)', letterSpacing: '-0.02em',
        }}>
          Contactless Vital Measurement
        </h1>
        <p style={{ color: 'var(--text2)', marginTop: '0.4rem', fontSize: '0.9rem' }}>
          PulseRoute · Non-contact photoplethysmography
        </p>
      </div>

      {/* Main card */}
      <div style={{
        width: '100%', maxWidth: 480,
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        boxShadow: 'var(--shadow-md)',
        overflow: 'hidden',
      }}>

        {/* Progress bar at top */}
        <div style={{ height: 4, background: '#e5e7eb' }}>
          <div style={{
            height: '100%', width: `${pct}%`,
            background: isComplete ? 'var(--success)' : 'var(--accent)',
            transition: 'width 0.6s ease',
          }} />
        </div>

        <div style={{ padding: '2rem' }}>

          {/* Status + ring */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '1.5rem' }}>
            <div style={{ position: 'relative', flexShrink: 0 }}>
              {isRunning && (
                <div className="pulse-ring" style={{
                  position: 'absolute', inset: 0, borderRadius: '50%',
                  border: '2px solid var(--accent)',
                }} />
              )}
              <ProgressRing pct={pct} />
            </div>
            <div>
              <div style={{
                display: 'inline-flex', alignItems: 'center', gap: '6px',
                padding: '3px 10px', borderRadius: 999, fontSize: '12px',
                fontWeight: 500, marginBottom: '0.5rem',
                ...(isComplete
                  ? { background: 'var(--success-bg)', color: 'var(--success)' }
                  : isRunning
                    ? { background: 'var(--accent-bg)', color: 'var(--accent)' }
                    : { background: 'var(--surface2)', color: 'var(--text3)' }),
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%', display: 'inline-block',
                  background: isComplete ? 'var(--success)' : isRunning ? 'var(--accent)' : '#d1d5db',
                  ...(isRunning ? {} : {}),
                }} className={isRunning ? 'blink' : ''} />
                {isComplete ? 'Measurement complete' : isRunning ? 'Measuring…' : 'Waiting to start'}
              </div>
              {state.message && (
                <p style={{ fontSize: '0.85rem', color: 'var(--text2)', lineHeight: 1.4 }}>
                  {state.message}
                </p>
              )}
              {isRunning && (
                <p style={{ fontSize: '0.8rem', color: 'var(--text3)', marginTop: '0.25rem' }}>
                  Please look at the camera and stay still
                </p>
              )}
            </div>
          </div>

          {/* Motion warning */}
          {isRunning && state.motion_pct > 20 && (
            <div className="fade-up" style={{
              padding: '10px 14px', borderRadius: 'var(--radius-md)',
              background: 'var(--warning-bg)',
              border: '1px solid #fcd34d',
              fontSize: '0.83rem', color: 'var(--warning)',
              marginBottom: '1.2rem',
              display: 'flex', alignItems: 'center', gap: '8px',
            }}>
              <span>⚠</span>
              Movement detected — please hold still for accurate results
            </div>
          )}

          {/* Step checklist */}
          {(isRunning || isComplete) && (
            <div style={{
              borderTop: '1px solid var(--border)',
              paddingTop: '1.2rem', marginBottom: '1.2rem',
            }}>
              <p style={{
                fontSize: '11px', fontWeight: 500, color: 'var(--text3)',
                letterSpacing: '0.08em', textTransform: 'uppercase',
                marginBottom: '0.75rem',
              }}>
                Measurement steps
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {STEPS.map(({ id, label }) => {
                  const done = doneSteps.includes(id)
                  return (
                    <div key={id} style={{
                      display: 'flex', alignItems: 'center', gap: '10px',
                      fontSize: '0.83rem',
                      color: done ? 'var(--text)' : 'var(--text3)',
                      transition: 'color 0.3s',
                    }}>
                      <div style={{
                        width: 18, height: 18, borderRadius: '50%', flexShrink: 0,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '10px',
                        background: done ? 'var(--success-bg)' : 'var(--surface2)',
                        border: `1px solid ${done ? '#6ee7b7' : 'var(--border)'}`,
                        color: done ? 'var(--success)' : 'var(--text3)',
                        transition: 'all 0.3s',
                      }}>
                        {done ? '✓' : ''}
                      </div>
                      {label}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Final results */}
          {isComplete && final.hr_bpm && (
            <div className="fade-up" style={{
              borderTop: '1px solid var(--border)',
              paddingTop: '1.2rem',
            }}>
              <p style={{
                fontSize: '11px', fontWeight: 500, color: 'var(--text3)',
                letterSpacing: '0.08em', textTransform: 'uppercase',
                marginBottom: '1rem',
              }}>
                Your results
              </p>
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px',
                marginBottom: '1rem',
              }}>
                <div style={{
                  background: 'var(--accent-bg)',
                  border: '1px solid #bfdbfe',
                  borderRadius: 'var(--radius-md)',
                  padding: '1rem', textAlign: 'center',
                }}>
                  <p style={{ fontSize: '2rem', fontWeight: 600, color: 'var(--accent)', lineHeight: 1 }}>
                    {final.hr_bpm}
                  </p>
                  <p style={{ fontSize: '11px', color: '#3b82f6', marginTop: '4px' }}>BPM · Heart rate</p>
                </div>
                {final.rr_bpm && (
                  <div style={{
                    background: 'var(--surface2)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-md)',
                    padding: '1rem', textAlign: 'center',
                  }}>
                    <p style={{ fontSize: '2rem', fontWeight: 600, color: 'var(--text)', lineHeight: 1 }}>
                      {final.rr_bpm}
                    </p>
                    <p style={{ fontSize: '11px', color: 'var(--text3)', marginTop: '4px' }}>BrPM · Respiratory</p>
                  </div>
                )}
              </div>

              {final.route_palm && (
                <div style={{
                  padding: '12px 14px',
                  background: 'var(--warning-bg)',
                  border: '1px solid #fcd34d',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '0.83rem', color: 'var(--warning)',
                  display: 'flex', alignItems: 'flex-start', gap: '8px',
                }}>
                  <span style={{ flexShrink: 0 }}>⚠</span>
                  <span>
                    For a more accurate HRV reading, please show your palm to the camera when prompted.
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <p style={{
        marginTop: '1.5rem', fontSize: '11px', color: 'var(--text3)',
        fontFamily: 'var(--font-mono)',
      }}>
        For investigational use only · PulseRoute v1.0
      </p>
    </div>
  )
}