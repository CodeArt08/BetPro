import { useState, useEffect } from 'react'
import { 
  Brain, Zap, TrendingUp, Activity, 
  Shield, BarChart2, Cpu, Target, RefreshCw,
  ChevronDown, ChevronUp, Info
} from 'lucide-react'
import api from '../services/api'
import { useRefreshOnUpdate } from '../hooks/useRealTimeUpdates'

// ─── Types ───────────────────────────────────────────────────────────
interface EngineSignals {
  cycle?: {
    V?: { rate_10: number; overdue_score: number; overdue: boolean; saturated: boolean }
    N?: { rate_10: number; overdue_score: number; overdue: boolean; saturated: boolean }
    D?: { rate_10: number; overdue_score: number; overdue: boolean; saturated: boolean }
  }
  streak?: {
    V?: { current_streak: number; correction_prob: number; correction_imminent: boolean }
    N?: { current_streak: number; correction_prob: number; correction_imminent: boolean }
    D?: { current_streak: number; correction_prob: number; correction_imminent: boolean }
  }
  fourier?: {
    cycle_detected: boolean
    cycle_length: number | null
    phase: number | null
    dominant_type: string | null
    dominant_amplitude: number
  }
  autocorr?: {
    lag1?: { autocorr: number; pattern: string }
    lag2?: { autocorr: number; pattern: string }
    lag3?: { autocorr: number; pattern: string }
  }
  runs_test?: {
    z_score: number
    p_value: number
    random: boolean
    exploitable: boolean
    reduce_stakes: boolean
  }
  changepoint?: {
    recent: any | null
    in_last_10: boolean
  }
  symbolic?: {
    pattern?: string
    lift?: number
    next_type?: string
    next_conf?: number
    exploitable?: boolean
  }
  shin?: {
    additive: Record<string, number>
    power: Record<string, number>
    shin: Record<string, number>
    bookmaker_margin: number
    anomaly: boolean
    divergence_score: number
  }
  kl?: number
  regime?: string
  engine_scores?: { V: number; N: number; D: number }
  odds_movement?: { signal?: { type: string } }
}

interface DashboardData {
  engine_state?: {
    draw_rate_10: number
    home_rate_10: number
    kl_divergence: number
    regime: string
    fourier: any
    changepoint: any
  }
  active_biases?: {
    draw_overdue: number
    streak: any
    fourier_signal: string | null
    symbolic_top: string
  }
  learning?: {
    active_lessons: number
    recovery_mode: boolean
    consecutive_errors: number
    ece: number
    changepoint_risk: string
  }
  performance?: {
    inference_avg: number
    inference_max: number
    fast_mode_today: number
    cache_ready: boolean
  }
  bankroll?: {
    bankroll: number
    roi_pct: number
    win_rate: number
    wins: number
    losses: number
    drawdown_pct: number
    is_stopped: boolean
    wins_streak: number
  }
}

// ─── Utils ───────────────────────────────────────────────────────────
const pct = (v: number) => `${(v * 100).toFixed(1)}%`
const pct0 = (v: number) => `${(v * 100).toFixed(0)}%`
const score = (v: number) => v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3)
const clamp01 = (v: number) => Math.max(0, Math.min(1, v))

const OUTCOME_LABELS: Record<string, string> = { V: '1 (Domicile)', N: 'X (Nul)', D: '2 (Extérieur)' }
const OUTCOME_COLORS: Record<string, string> = {
  V: 'text-green-400',
  N: 'text-yellow-400',
  D: 'text-red-400',
}
const OUTCOME_BG: Record<string, string> = {
  V: 'bg-green-600',
  N: 'bg-yellow-600',
  D: 'bg-red-600',
}

const RegimeBadge = ({ regime }: { regime?: string }) => {
  const color = regime === 'STABLE' ? 'bg-green-900 text-green-300 border-green-600'
    : regime === 'VOLATILE' ? 'bg-yellow-900 text-yellow-300 border-yellow-600'
    : 'bg-red-900 text-red-300 border-red-600'
  return <span className={`px-2 py-0.5 rounded text-xs font-bold border ${color}`}>{regime || '?'}</span>
}

// ─── Signal Bar ───────────────────────────────────────────────────────
const SignalBar = ({ value, max = 1, color = 'bg-blue-500', label }: {
  value: number; max?: number; color?: string; label?: string
}) => {
  const pctWidth = clamp01(Math.abs(value) / max) * 100
  const isNeg = value < 0
  return (
    <div className="flex items-center gap-2 text-xs">
      {label && <span className="text-gray-400 w-20 shrink-0">{label}</span>}
      <div className="flex-1 bg-gray-700 rounded-full h-2 relative">
        <div
          className={`h-2 rounded-full ${isNeg ? 'bg-red-500' : color}`}
          style={{ width: `${pctWidth}%` }}
        />
      </div>
      <span className="text-gray-300 w-12 text-right">{score(value)}</span>
    </div>
  )
}

// ─── Section Component ────────────────────────────────────────────────
const Section = ({ title, icon: Icon, children, defaultOpen = true, badge }: {
  title: string; icon: any; children: React.ReactNode; defaultOpen?: boolean; badge?: React.ReactNode
}) => {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="bg-gray-800/60 border border-gray-700 rounded-xl overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-700/50 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-blue-400" />
          <span className="font-semibold text-sm">{title}</span>
          {badge}
        </div>
        {open ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
      </button>
      {open && <div className="px-4 pb-4 space-y-2">{children}</div>}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────
export default function EngineSignalsPanel() {
  const [signals, setSignals] = useState<EngineSignals | null>(null)
  const [dashboard, setDashboard] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [cacheReady, setCacheReady] = useState(false)

  const loadData = async () => {
    try {
      const [sigData, dashData] = await Promise.all([
        api.get('/api/realtime/signals').then(r => r.data).catch(() => null),
        api.get('/api/realtime/dashboard').then(r => r.data).catch(() => null),
      ])
      // Only update signals if we got valid data with required fields
      if (sigData && sigData.cycle && sigData.streak) {
        setSignals(sigData)
        setCacheReady(true)
      } else if (!cacheReady) {
        // Only set cacheReady to false if we never had valid data
        // This prevents flickering when API temporarily fails
        console.log('Signals data incomplete, keeping previous state')
      }
      if (dashData) setDashboard(dashData)
      setLastUpdate(new Date())
    } catch {
      // silent - keep previous state on error
    } finally {
      setLoading(false)
    }
  }

  // Refresh on result entry (event-driven via socket)
  useRefreshOnUpdate(loadData)
  
  // Initial load
  useEffect(() => { loadData() }, [])

  if (loading && !signals) {
    return (
      <div className="bg-gray-800/40 border border-gray-700 rounded-xl p-6 text-center text-gray-400">
        <Cpu className="w-8 h-8 mb-2 mx-auto animate-pulse text-blue-400" />
        Chargement du moteur temps réel...
      </div>
    )
  }

  // Show loading state only on initial load
  if (loading && !signals && !cacheReady) {
    return (
      <div className="bg-gray-800/40 border border-gray-700 rounded-xl p-4 text-center text-gray-500 text-sm">
        <Cpu className="w-5 h-5 mb-1 mx-auto animate-pulse text-blue-400" />
        Initialisation du moteur temps réel...
      </div>
    )
  }

  // Only show "not available" if we never got valid data AND loading is complete
  if (!signals && !cacheReady && !loading) {
    return (
      <div className="bg-gray-800/40 border border-gray-700 rounded-xl p-4 text-center text-gray-500 text-sm">
        <Info className="w-5 h-5 mb-1 mx-auto" />
        Cache moteur non disponible — données insuffisantes
        <button 
          onClick={loadData}
          className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs text-white"
        >
          Réessayer
        </button>
      </div>
    )
  }

  const cycle = signals?.cycle || {}
  const streak = signals?.streak || {}
  const fourier = signals?.fourier
  const autocorr = signals?.autocorr || {}
  const runs = signals?.runs_test
  const changepoint = signals?.changepoint
  const symbolic = signals?.symbolic
  const shin = signals?.shin
  const engineScores = signals?.engine_scores || { V: 0, N: 0, D: 0 }
  const regime = signals?.regime || dashboard?.engine_state?.regime || 'STABLE'
  const bankroll = dashboard?.bankroll
  const learning = dashboard?.learning
  const perf = dashboard?.performance

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <span className="font-bold text-sm">Moteur Temps Réel</span>
          <RegimeBadge regime={regime} />
        </div>
        <div className="flex items-center gap-2">
          {lastUpdate && (
            <span className="text-xs text-gray-500">
              {lastUpdate.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          )}
          <button
            onClick={loadData}
            className="p-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5 text-gray-300" />
          </button>
        </div>
      </div>

      {/* Engine Scores */}
      <Section title="Scores Engine Composites" icon={Target}>
        <div className="grid grid-cols-3 gap-2 mb-2">
          {(['V', 'N', 'D'] as const).map(t => (
            <div key={t} className="bg-gray-900/60 rounded-lg p-2 text-center">
              <div className={`text-xs font-medium ${OUTCOME_COLORS[t]}`}>{OUTCOME_LABELS[t]}</div>
              <div className={`text-lg font-bold mt-1 ${engineScores[t as keyof typeof engineScores] > 0.05 ? 'text-green-400' : engineScores[t as keyof typeof engineScores] < -0.05 ? 'text-red-400' : 'text-gray-300'}`}>
                {score(engineScores[t as keyof typeof engineScores] || 0)}
              </div>
            </div>
          ))}
        </div>
        <div className="space-y-1 mt-2">
          {(['V', 'N', 'D'] as const).map(t => (
            <SignalBar
              key={t}
              label={t}
              value={engineScores[t as keyof typeof engineScores] || 0}
              max={0.5}
              color={OUTCOME_BG[t].replace('bg-', 'bg-')}
            />
          ))}
        </div>
      </Section>

      {/* Cycle Analysis */}
      <Section title="M2 — Cycles & Fréquences" icon={Activity}>
        <div className="grid grid-cols-3 gap-2">
          {(['V', 'N', 'D'] as const).map(t => {
            const c = cycle[t] || {}
            return (
              <div key={t} className={`rounded-lg p-2 border ${(c as any).overdue ? 'border-orange-500/50 bg-orange-900/10' : 'border-gray-700 bg-gray-900/40'}`}>
                <div className={`text-xs font-bold ${OUTCOME_COLORS[t]}`}>{t}</div>
                <div className="text-xs text-gray-400 mt-1">
                  Taux 10: <span className="text-white">{pct0((c as any).rate_10 || 0)}</span>
                </div>
                <div className="text-xs text-gray-400">
                  Overdue: <span className={`font-bold ${(c as any).overdue ? 'text-orange-400' : 'text-gray-400'}`}>
                    {pct0((c as any).overdue_score || 0)}
                  </span>
                </div>
                {(c as any).overdue && (
                  <div className="text-xs text-orange-300 mt-1 font-medium">⚠ OVERDUE</div>
                )}
                {(c as any).saturated && (
                  <div className="text-xs text-red-300 mt-1 font-medium">🔴 SATURÉ</div>
                )}
              </div>
            )
          })}
        </div>
      </Section>

      {/* Streak Analysis */}
      <Section title="M3 — Analyse Streaks" icon={TrendingUp}>
        <div className="grid grid-cols-3 gap-2">
          {(['V', 'N', 'D'] as const).map(t => {
            const s = streak[t] || {}
            return (
              <div key={t} className={`rounded-lg p-2 border ${(s as any).correction_imminent ? 'border-red-500/50 bg-red-900/10' : 'border-gray-700 bg-gray-900/40'}`}>
                <div className={`text-xs font-bold ${OUTCOME_COLORS[t]}`}>{t}</div>
                <div className="text-xl font-bold text-center mt-1">
                  {(s as any).current_streak || 0}
                </div>
                <div className="text-xs text-gray-400 text-center">
                  Correct. prob: {pct0((s as any).correction_prob || 0)}
                </div>
                {(s as any).correction_imminent && (
                  <div className="text-xs text-red-300 text-center mt-1 font-medium">CORRECTION</div>
                )}
              </div>
            )
          })}
        </div>
      </Section>

      {/* Fourier + BOCPD */}
      <Section
        title="M5/M6 — Fourier & Changepoint"
        icon={Zap}
        badge={fourier?.cycle_detected
          ? <span className="ml-2 text-xs bg-purple-900 text-purple-300 border border-purple-600 px-1.5 py-0.5 rounded">CYCLE DÉTECTÉ</span>
          : undefined
        }
      >
        <div className="grid grid-cols-2 gap-3">
          {/* Fourier */}
          <div className="bg-gray-900/40 rounded-lg p-3 border border-gray-700">
            <div className="text-xs text-gray-400 mb-2 font-medium">FFT Fourier</div>
            {fourier?.cycle_detected ? (
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Cycle:</span>
                  <span className="text-purple-300 font-bold">{fourier.cycle_length} matchs</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Phase:</span>
                  <span className="text-white">{fourier.phase} / {fourier.cycle_length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Signal:</span>
                  <span className={`font-bold ${OUTCOME_COLORS[fourier.dominant_type || 'V']}`}>
                    {OUTCOME_LABELS[fourier.dominant_type || 'V'] || '?'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-xs text-gray-500 text-center py-2">Pas de cycle détecté</div>
            )}
          </div>

          {/* BOCPD */}
          <div className={`bg-gray-900/40 rounded-lg p-3 border ${changepoint?.in_last_10 ? 'border-red-500/40' : 'border-gray-700'}`}>
            <div className="text-xs text-gray-400 mb-2 font-medium">BOCPD Changepoint</div>
            {changepoint?.in_last_10 ? (
              <div className="space-y-1 text-xs">
                <div className="text-red-300 font-bold">⚡ RUPTURE RÉCENTE</div>
                <div className="text-gray-400">Modifier confiance ×0.75</div>
                <div className="text-gray-400">Risk: ÉLEVÉ</div>
              </div>
            ) : (
              <div className="space-y-1 text-xs">
                <div className="text-green-300">✓ Stable</div>
                <div className="text-gray-400">Pas de rupture récente</div>
                <div className="text-gray-400">Risk: {learning?.changepoint_risk || 'FAIBLE'}</div>
              </div>
            )}
          </div>
        </div>
      </Section>

      {/* Runs Test + Autocorr */}
      <Section title="M4/M7 — Autocorrélation & Runs Test" icon={BarChart2}>
        <div className="grid grid-cols-2 gap-3">
          {/* Runs Test */}
          <div className={`bg-gray-900/40 rounded-lg p-3 border ${runs?.exploitable ? 'border-green-500/40' : runs?.random === false ? 'border-yellow-500/40' : 'border-gray-700'}`}>
            <div className="text-xs text-gray-400 mb-2 font-medium">Wald-Wolfowitz</div>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">Z-score:</span>
                <span className="text-white">{runs?.z_score?.toFixed(2) || '?'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">p-value:</span>
                <span className={`font-bold ${runs?.exploitable ? 'text-green-400' : runs?.reduce_stakes ? 'text-red-400' : 'text-gray-300'}`}>
                  {runs?.p_value?.toFixed(3) || '?'}
                </span>
              </div>
              <div className="mt-1">
                {runs?.exploitable && <span className="text-green-300 font-bold">EXPLOITABLE</span>}
                {runs?.reduce_stakes && <span className="text-red-300 font-bold">REDUCE STAKES</span>}
                {runs?.random && !runs?.reduce_stakes && <span className="text-gray-400">Aléatoire (normal)</span>}
              </div>
            </div>
          </div>

          {/* Autocorr */}
          <div className="bg-gray-900/40 rounded-lg p-3 border border-gray-700">
            <div className="text-xs text-gray-400 mb-2 font-medium">Autocorrélation</div>
            <div className="space-y-1">
              {['lag1', 'lag2', 'lag3'].map(lag => {
                const ac = (autocorr as any)[lag] || {}
                return (
                  <div key={lag} className="flex items-center gap-2 text-xs">
                    <span className="text-gray-400 w-10">{lag}:</span>
                    <div className="flex-1 bg-gray-700 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${(ac.autocorr || 0) > 0 ? 'bg-blue-500' : 'bg-orange-500'}`}
                        style={{ width: `${clamp01(Math.abs(ac.autocorr || 0)) * 100}%` }}
                      />
                    </div>
                    <span className="text-gray-300 w-14 text-right">{ac.autocorr?.toFixed(3) || '0.000'}</span>
                    <span className="text-gray-500 w-20">{ac.pattern || '?'}</span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </Section>

      {/* Symbolic Pattern */}
      {symbolic?.pattern && (
        <Section title="M8 — Pattern Symbolique (PrefixSpan)" icon={Brain}>
          <div className={`bg-gray-900/40 rounded-lg p-3 border ${symbolic.exploitable ? 'border-purple-500/40' : 'border-gray-700'}`}>
            <div className="flex flex-wrap items-center gap-3 text-xs">
              <div>
                <span className="text-gray-400">Pattern: </span>
                <span className="font-mono text-purple-300 text-lg font-bold">
                  {(symbolic.pattern || '').split('').join(' → ')}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Lift: </span>
                <span className={`font-bold ${(symbolic.lift || 0) > 1.5 ? 'text-green-400' : 'text-yellow-400'}`}>
                  ×{symbolic.lift?.toFixed(2) || '?'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Suivant: </span>
                <span className={`font-bold ${OUTCOME_COLORS[symbolic.next_type || 'V']}`}>
                  {OUTCOME_LABELS[symbolic.next_type || 'V']} ({pct0(symbolic.next_conf || 0)})
                </span>
              </div>
              {symbolic.exploitable && (
                <span className="px-2 py-0.5 bg-purple-900 text-purple-300 border border-purple-600 rounded text-xs font-bold">
                  EXPLOITABLE
                </span>
              )}
            </div>
          </div>
        </Section>
      )}

      {/* Shin Probabilities */}
      {shin && (
        <Section
          title="M12 — Décomposition Shin"
          icon={BarChart2}
          badge={shin.anomaly
            ? <span className="ml-2 text-xs bg-orange-900 text-orange-300 border border-orange-600 px-1.5 py-0.5 rounded">ANOMALIE</span>
            : undefined
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-400 border-b border-gray-700">
                  <th className="pb-1 text-left">Méthode</th>
                  <th className="pb-1 text-center">1 (Dom)</th>
                  <th className="pb-1 text-center">X (Nul)</th>
                  <th className="pb-1 text-center">2 (Ext)</th>
                </tr>
              </thead>
              <tbody className="space-y-1">
                {(['additive', 'power', 'shin'] as const).map(method => (
                  <tr key={method} className="border-b border-gray-700/40">
                    <td className="py-1.5 font-medium capitalize text-gray-300">{method}</td>
                    {(['V', 'N', 'D'] as const).map(t => (
                      <td key={t} className={`py-1.5 text-center font-medium ${OUTCOME_COLORS[t]}`}>
                        {pct0(shin[method]?.[t] || 0)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex flex-wrap gap-3 mt-2 text-xs text-gray-400">
            <span>Marge bookmaker: <span className="text-white">{pct(shin.bookmaker_margin || 0)}</span></span>
            <span>Div score: <span className={`${shin.anomaly ? 'text-orange-400 font-bold' : 'text-gray-300'}`}>
              {shin.divergence_score?.toFixed(3) || '0.000'}
            </span></span>
          </div>
        </Section>
      )}

      {/* Learning Status */}
      {learning && (
        <Section
          title="Apprentissage & Robustesse"
          icon={Shield}
          badge={learning.recovery_mode
            ? <span className="ml-2 text-xs bg-red-900 text-red-300 border border-red-600 px-1.5 py-0.5 rounded animate-pulse">RECOVERY</span>
            : undefined
          }
        >
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className={`rounded-lg p-2.5 border ${learning.recovery_mode ? 'border-red-500/50 bg-red-900/10' : 'border-gray-700 bg-gray-900/40'}`}>
              <div className="text-gray-400">Mode</div>
              <div className={`font-bold ${learning.recovery_mode ? 'text-red-300' : 'text-green-300'}`}>
                {learning.recovery_mode ? 'RECOVERY' : 'NORMAL'}
              </div>
            </div>
            <div className="rounded-lg p-2.5 border border-gray-700 bg-gray-900/40">
              <div className="text-gray-400">Erreurs consécutives</div>
              <div className={`font-bold ${(learning.consecutive_errors || 0) >= 3 ? 'text-red-400' : 'text-gray-200'}`}>
                {learning.consecutive_errors || 0}
              </div>
            </div>
            <div className={`rounded-lg p-2.5 border ${(learning.ece || 0) > 0.07 ? 'border-orange-500/50 bg-orange-900/10' : 'border-gray-700 bg-gray-900/40'}`}>
              <div className="text-gray-400">ECE (calibration)</div>
              <div className={`font-bold ${(learning.ece || 0) > 0.12 ? 'text-red-400' : (learning.ece || 0) > 0.07 ? 'text-orange-400' : 'text-green-400'}`}>
                {learning.ece?.toFixed(3) || '0.000'}
              </div>
            </div>
            <div className="rounded-lg p-2.5 border border-gray-700 bg-gray-900/40">
              <div className="text-gray-400">Leçons actives</div>
              <div className="font-bold text-blue-300">{learning.active_lessons || 0}</div>
            </div>
          </div>
        </Section>
      )}

      {/* Bankroll */}
      {bankroll && (
        <Section
          title="Bankroll & ROI"
          icon={TrendingUp}
          badge={bankroll.is_stopped
            ? <span className="ml-2 text-xs bg-red-900 text-red-300 border border-red-600 px-1.5 py-0.5 rounded">STOP</span>
            : undefined
          }
        >
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg p-2.5 border border-gray-700 bg-gray-900/40">
              <div className="text-gray-400">Bankroll</div>
              <div className="font-bold text-white text-base">
                {bankroll.bankroll?.toLocaleString('fr-FR')} Ar
              </div>
            </div>
            <div className="rounded-lg p-2.5 border border-gray-700 bg-gray-900/40">
              <div className="text-gray-400">ROI</div>
              <div className={`font-bold text-base ${(bankroll.roi_pct || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {(bankroll.roi_pct || 0) >= 0 ? '+' : ''}{bankroll.roi_pct?.toFixed(1) || '0.0'}%
              </div>
            </div>
            <div className={`rounded-lg p-2.5 border ${(bankroll.drawdown_pct || 0) > 15 ? 'border-red-500/50 bg-red-900/10' : (bankroll.drawdown_pct || 0) > 8 ? 'border-orange-500/40' : 'border-gray-700 bg-gray-900/40'}`}>
              <div className="text-gray-400">Drawdown</div>
              <div className={`font-bold ${(bankroll.drawdown_pct || 0) > 15 ? 'text-red-400' : (bankroll.drawdown_pct || 0) > 8 ? 'text-orange-400' : 'text-green-400'}`}>
                {bankroll.drawdown_pct?.toFixed(1) || '0.0'}%
              </div>
            </div>
            <div className="rounded-lg p-2.5 border border-gray-700 bg-gray-900/40">
              <div className="text-gray-400">Win Rate</div>
              <div className="font-bold text-blue-300">
                {pct0(bankroll.win_rate || 0)} ({bankroll.wins || 0}W / {bankroll.losses || 0}L)
              </div>
            </div>
          </div>

          {/* Drawdown mini-bar */}
          <div className="mt-2">
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Drawdown courant</span>
              <span>{bankroll.drawdown_pct?.toFixed(1)}% / 20%</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full">
              <div
                className={`h-2 rounded-full transition-all ${
                  (bankroll.drawdown_pct || 0) > 15 ? 'bg-red-500' :
                  (bankroll.drawdown_pct || 0) > 8 ? 'bg-orange-500' : 'bg-green-500'
                }`}
                style={{ width: `${Math.min((bankroll.drawdown_pct || 0) / 20 * 100, 100)}%` }}
              />
            </div>
          </div>
        </Section>
      )}

      {/* Performance */}
      {perf && (
        <Section title="Performance Inference" icon={Cpu} defaultOpen={false}>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg p-2.5 border border-gray-700 bg-gray-900/40">
              <div className="text-gray-400">Latence moy.</div>
              <div className={`font-bold ${(perf.inference_avg || 0) > 6000 ? 'text-red-400' : 'text-green-400'}`}>
                {perf.inference_avg?.toFixed(0) || '?'} ms
              </div>
            </div>
            <div className="rounded-lg p-2.5 border border-gray-700 bg-gray-900/40">
              <div className="text-gray-400">Latence max.</div>
              <div className={`font-bold ${(perf.inference_max || 0) > 8000 ? 'text-red-400' : 'text-gray-200'}`}>
                {perf.inference_max?.toFixed(0) || '?'} ms
              </div>
            </div>
            <div className="rounded-lg p-2.5 border border-gray-700 bg-gray-900/40">
              <div className="text-gray-400">Fast Mode (auj.)</div>
              <div className={`font-bold ${(perf.fast_mode_today || 0) > 3 ? 'text-orange-400' : 'text-gray-200'}`}>
                {perf.fast_mode_today || 0}×
              </div>
            </div>
            <div className={`rounded-lg p-2.5 border ${perf.cache_ready ? 'border-green-500/40' : 'border-red-500/40'} bg-gray-900/40`}>
              <div className="text-gray-400">Cache</div>
              <div className={`font-bold ${perf.cache_ready ? 'text-green-400' : 'text-red-400'}`}>
                {perf.cache_ready ? '✓ PRÊT' : '✗ VIDE'}
              </div>
            </div>
          </div>
        </Section>
      )}
    </div>
  )
}
