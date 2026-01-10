/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Outfit', 'system-ui', 'sans-serif'],
      },
      colors: {
        observatory: {
          bg: '#0a0f1a',
          card: '#111827',
          border: '#1e293b',
          accent: '#22d3ee',
          success: '#4ade80',
          warning: '#fbbf24',
          error: '#f87171',
        }
      }
    },
  },
  plugins: [],
}

