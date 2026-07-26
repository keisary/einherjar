import { useEffect, useRef, useCallback } from 'react'

const RUNES = 'ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟ'

interface Particle {
  x: number
  y: number
  char: string
  vx: number
  vy: number
  opacity: number
  life: number
  maxLife: number
  size: number
}

export function RuneCrumble({
  density = 'medium',
  className = '',
}: {
  density?: 'low' | 'medium' | 'high'
  className?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const particlesRef = useRef<Particle[]>([])
  const rafRef = useRef<number>(0)
  const spawnRateRef = useRef(0)

  const spawnRate = density === 'low' ? 0.3 : density === 'medium' ? 0.6 : 1.2

  const spawnParticle = useCallback(
    (w: number, h: number) => {
      const side = Math.floor(Math.random() * 4)
      let x: number, y: number
      switch (side) {
        case 0:
          x = Math.random() * w
          y = -10
          break
        case 1:
          x = w + 10
          y = Math.random() * h
          break
        case 2:
          x = Math.random() * w
          y = h + 10
          break
        default:
          x = -10
          y = Math.random() * h
          break
      }
      const angle = Math.atan2(h / 2 - y, w / 2 - x) + (Math.random() - 0.5) * 1.5
      const speed = 0.3 + Math.random() * 0.8
      const life = 60 + Math.random() * 120
      return {
        x,
        y,
        char: RUNES[Math.floor(Math.random() * RUNES.length)],
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        opacity: 0,
        life,
        maxLife: life,
        size: 10 + Math.random() * 14,
      }
    },
    []
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect()
      if (rect) {
        canvas.width = rect.width
        canvas.height = rect.height
      }
    }
    resize()
    window.addEventListener('resize', resize)

    const animate = () => {
      if (!ctx || !canvas) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      spawnRateRef.current += spawnRate
      while (spawnRateRef.current >= 1) {
        spawnRateRef.current -= 1
        particlesRef.current.push(spawnParticle(canvas.width, canvas.height))
      }

      const particles = particlesRef.current
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i]
        p.x += p.vx
        p.y += p.vy
        p.life--

        // Fade in then out
        const progress = 1 - p.life / p.maxLife
        if (progress < 0.15) {
          p.opacity = progress / 0.15
        } else if (progress > 0.7) {
          p.opacity = (1 - progress) / 0.3
        } else {
          p.opacity = 1
        }

        if (p.life <= 0) {
          particles.splice(i, 1)
          continue
        }

        ctx.save()
        ctx.globalAlpha = p.opacity * 0.6
        ctx.fillStyle = '#b4c8dc'
        ctx.font = `${p.size}px serif`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(p.char, p.x, p.y)
        ctx.restore()
      }

      rafRef.current = requestAnimationFrame(animate)
    }

    rafRef.current = requestAnimationFrame(animate)
    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', resize)
    }
  }, [spawnParticle, spawnRate])

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{ zIndex: 0 }}
    />
  )
}
