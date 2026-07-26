/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#050505',
        surface: '#0A0A0A',
        surfaceHighlight: '#111111',
        border: '#1A1A1A',
        borderHighlight: '#2A2A2A',
        textPrimary: '#E8E8E8',
        textSecondary: '#8A8A8A',
        textMuted: '#4A4A4A',
        frost: '#B0C4DE',
        frostDark: '#6B7B8D',
        ice: '#A5C9D4',
        mist: '#7B8B9B',
        positive: '#5C8D6F',
        negative: '#8B4A4A',
        warning: '#8B7A3D',
      },
      fontFamily: {
        cinzel: ['Cinzel', 'Times New Roman', 'serif'],
        inter: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Courier New', 'monospace'],
      },
      letterSpacing: {
        widest: '0.2em',
      },
      borderRadius: {
        none: '0',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'count': 'count 0.5s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(176, 196, 222, 0.1)' },
          '100%': { boxShadow: '0 0 20px rgba(176, 196, 222, 0.2)' },
        },
      },
    },
  },
  plugins: [],
}
