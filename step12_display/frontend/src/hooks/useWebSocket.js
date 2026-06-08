import { useEffect, useRef, useState, useCallback } from 'react'

const WS_BASE = 'ws://localhost:8000'
const RECONNECT_DELAY_MS = 2500

/**
 * useWebSocket — connects to the PulseRoute dashboard server.
 *
 * Handles two message types:
 *   1. State events   — merged into `state` and tracked in `snrHistory`
 *   2. Frame messages — { type: "frame", frame: "<base64 JPEG>" }
 *                       stored in `latestFrame` as a data URL
 *
 * Returns
 * -------
 * state       : current pipeline state object
 * connected   : boolean WebSocket connection status
 * snrHistory  : array of { t, snr } for the SNR chart
 * latestFrame : data URL string of the latest annotated frame,
 *               or null if no frame received yet
 */
export function useWebSocket(path) {
  const [state,       setState]       = useState({
    status: 'idle', progress: 0, steps: {},
  })
  const [connected,   setConnected]   = useState(false)
  const [snrHistory,  setSnrHistory]  = useState([])
  const [latestFrame, setLatestFrame] = useState(null)

  const wsRef      = useRef(null)
  const retryTimer = useRef(null)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return

    const ws = new WebSocket(`${WS_BASE}${path}`)
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) return
      setConnected(true)
    }

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return
      try {
        const msg = JSON.parse(evt.data)

        // ── Frame message ────────────────────────
        if (msg.type === 'frame' && msg.frame) {
          setLatestFrame(`data:image/jpeg;base64,${msg.frame}`)
          return
        }

        // ── Ping (keepalive) ─────────────────────
        if (msg.ping) return

        // ── State event ──────────────────────────
        setState(prev => {
          const next = {
            ...prev,
            ...msg,
            steps: {
              ...prev.steps,
              ...(msg.steps || {}),
            },
          }
          return next
        })

        // Track SNR score history for chart
        const snr = msg.snr_score
          ?? msg.final?.snr_score
          ?? null
        if (snr != null) {
          setSnrHistory(prev => [
            ...prev.slice(-59),   // keep last 60 points
            { t: Date.now(), snr: parseFloat(snr) },
          ])
        }

        // Clear frame when pipeline resets to idle
        if (msg.status === 'idle') {
          setLatestFrame(null)
          setSnrHistory([])
        }

      } catch {
        // Ignore malformed messages
      }
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      setConnected(false)
      retryTimer.current = setTimeout(connect, RECONNECT_DELAY_MS)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [path])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      clearTimeout(retryTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { state, connected, snrHistory, latestFrame }
}