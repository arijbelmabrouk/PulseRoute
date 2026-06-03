import { useState, useEffect, useRef, useCallback } from 'react'

export function useWebSocket(endpoint) {
  const [state, setState]         = useState({ status: 'idle', progress: 0, steps: {} })
  const [connected, setConnected] = useState(false)
  const [snrHistory, setSnrHistory] = useState([])
  const wsRef           = useRef(null)
  const reconnectTimer  = useRef(null)

  const connect = useCallback(() => {
    // Vite proxy forwards /ws → ws://localhost:8000
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host     = window.location.host
    const url      = `${protocol}://${host}${endpoint}`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      clearTimeout(reconnectTimer.current)
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        // Ignore keep-alive pings
        if (data.ping) return

        setState(prev => ({
          ...prev,
          ...data,
          steps: { ...prev.steps, ...(data.steps || {}) },
        }))

        // Track SNR history for the doctor chart
        if (data.snr_score !== undefined) {
          setSnrHistory(h => [
            ...h.slice(-59),
            { t: h.length, snr: parseFloat(data.snr_score.toFixed(3)) }
          ])
        }
        // Also track from final results
        if (data.final?.snr_score !== undefined) {
          setSnrHistory(h => [
            ...h.slice(-59),
            { t: h.length, snr: parseFloat(data.final.snr_score.toFixed(3)) }
          ])
        }
      } catch (_) { /* ignore parse errors */ }
    }

    ws.onclose = () => {
      setConnected(false)
      // Reconnect after 3 seconds
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [endpoint])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { state, connected, snrHistory }
}
