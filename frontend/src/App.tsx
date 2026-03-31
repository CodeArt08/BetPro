import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Dashboard from './pages/Dashboard'
import Predictions from './pages/Predictions'
import Matches from './pages/Matches'
import Teams from './pages/Teams'
import PredictionComparisonModal from './components/PredictionComparisonModal'
import { LayoutDashboard, Brain, Calendar, Users, Bell, X } from 'lucide-react'
import socketService from './services/socket'

interface PredictionResult {
  match: string
  matchday: number
  predicted: string
  actual: string
  actual_result: string | null
  is_correct: boolean
  stake: number
  profit_loss: number
  score: string | null
}

interface ComparisonData {
  predictions: PredictionResult[]
  total_stake: number
  total_profit: number
  correct_count: number
  total_predictions: number
  accuracy: number
  timestamp: string
}

function App() {
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null)
  const [showModal, setShowModal] = useState(false)
  const [notification, setNotification] = useState<{show: boolean; message: string; count: number}>({show: false, message: '', count: 0})

  useEffect(() => {
    // Connect to socket and listen for prediction comparison events
    socketService.connect()
    
    const unsub1 = socketService.subscribe('prediction_comparison', (data: ComparisonData) => {
      console.log('Received prediction comparison:', data)
      setComparisonData(data)
      setShowModal(true)
    })

    // Listen for new predictions generated
    const unsub2 = socketService.subscribe('prediction_generated', (data: {match_id: number; prediction_id: number; predicted_result: string}) => {
      console.log('New prediction generated:', data)
      setNotification(prev => ({
        show: true,
        message: 'Nouvelle prédiction générée!',
        count: prev.count + 1
      }))
    })

    // Auto-hide notification after 5 seconds
    let timeoutId: ReturnType<typeof setTimeout>
    if (notification.show) {
      timeoutId = setTimeout(() => {
        setNotification({show: false, message: '', count: 0})
      }, 5000)
    }

    return () => {
      unsub1()
      unsub2()
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [notification.show])

  const navItems = [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/predictions', icon: Brain, label: 'Predictions' },
    { to: '/matches', icon: Calendar, label: 'Matches' },
    { to: '/teams', icon: Users, label: 'Teams' },
  ]

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-900">
        {/* Navigation */}
        <nav className="bg-gray-800 border-b border-gray-700">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-2">
                <Brain className="w-8 h-8 text-blue-500" />
                <span className="text-xl font-bold">Bet261 Engine</span>
              </div>
              <div className="flex space-x-1">
                {navItems.map(({ to, icon: Icon, label }) => (
                  <NavLink
                    key={to}
                    to={to}
                    className={({ isActive }) =>
                      `flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                        isActive
                          ? 'bg-blue-600 text-white'
                          : 'text-gray-300 hover:bg-gray-700'
                      }`
                    }
                  >
                    <Icon className="w-5 h-5" />
                    <span>{label}</span>
                  </NavLink>
                ))}
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 py-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/matches" element={<Matches />} />
            <Route path="/teams" element={<Teams />} />
          </Routes>
        </main>

        {/* Notification Toast */}
        {notification.show && (
          <div className="fixed bottom-4 right-4 bg-blue-600 text-white px-4 py-3 rounded-lg shadow-lg flex items-center space-x-3 z-50 animate-pulse">
            <Bell className="w-5 h-5" />
            <span className="font-medium">{notification.message}</span>
            {notification.count > 1 && (
              <span className="bg-blue-800 px-2 py-0.5 rounded-full text-sm">{notification.count}</span>
            )}
            <button 
              onClick={() => setNotification({show: false, message: '', count: 0})}
              className="ml-2 hover:bg-blue-700 rounded p-1"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Prediction Comparison Modal */}
        {showModal && (
          <PredictionComparisonModal 
            data={comparisonData} 
            onClose={() => setShowModal(false)} 
          />
        )}
      </div>
    </BrowserRouter>
  )
}

export default App
