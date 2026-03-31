import { useEffect } from 'react';
import socketService from '../services/socket';

export function useRealTimeUpdates(callbacks: Record<string, (data?: any) => void>) {
  useEffect(() => {
    socketService.connect();

    const unsubs = Object.entries(callbacks).map(([event, callback]) => {
      return socketService.subscribe(event, callback);
    });

    return () => {
      unsubs.forEach(unsub => unsub());
    };
  }, []); // Only connect once on mount
}

/**
 * Hook to force refresh when ANY real-time event occurs
 * Useful for simple "reload everything" behavior
 */
export function useRefreshOnUpdate(refreshCallback: () => void) {
  useEffect(() => {
    socketService.connect();
    
    // Listen to major data change events
    const unsubMatch = socketService.subscribe('match_completed', refreshCallback);
    const unsubUpcoming = socketService.subscribe('upcoming_matches_updated', refreshCallback);
    const unsubPrediction = socketService.subscribe('prediction_generated', refreshCallback);
    const unsubSeason = socketService.subscribe('season_created', (data: any) => {
      // If season created with reset_all flag, do a complete refresh
      if (data?.reset_all) {
        console.log('Season created with reset_all, forcing complete refresh');
        // Force complete page reload to reset all state
        window.location.reload();
      } else {
        refreshCallback();
      }
    });
    const unsubSelection = socketService.subscribe('dynamic_selection', refreshCallback);
    
    return () => {
      unsubMatch();
      unsubUpcoming();
      unsubPrediction();
      unsubSeason();
      unsubSelection();
    };
  }, [refreshCallback]);
}

/**
 * Hook specifically for season reset
 */
export function useSeasonReset(resetCallback: () => void) {
  useEffect(() => {
    socketService.connect();
    
    const unsubSeason = socketService.subscribe('season_created', (data: any) => {
      console.log('Season created event received:', data);
      if (data?.reset_all) {
        console.log('Season reset detected, resetting all data');
        resetCallback();
      } else {
        console.log('Season created but no reset flag');
      }
    });
    
    // Also listen to reset_all_data event for safety
    const unsubReset = socketService.subscribe('reset_all_data', (data: any) => {
      console.log('Reset all data event received:', data);
      resetCallback();
    });
    
    // Also listen to any event to debug
    const unsubAny = socketService.subscribe('any', (eventName: string, data: any) => {
      console.log('Socket event received:', eventName, data);
    });
    
    return () => {
      unsubSeason();
      unsubReset();
      unsubAny();
    };
  }, [resetCallback]);
}
