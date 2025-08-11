package com.nutrition.app.data

import com.nutrition.app.data.DateRange
import com.github.mikephil.charting.data.Entry
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class PlotsRepository @Inject constructor(
    private val plotsApiService: PlotsApiService
) {
    // No client-side fallback: if series are empty, return empty lists so the chart shows "Nothing found".
      // MPAndroidChart uses Float for X, which lacks precision for epoch milliseconds.
      // Use days-since-epoch as X to avoid precision loss and format back to full dates in the chart.

      private fun toDaysSinceEpoch(epochMs: Long): Float {
          return (epochMs / 86_400_000.0).toFloat()
      }

    suspend fun getWeightData(dateRange: DateRange): List<Entry> {
        val (simple, days) = when (dateRange) {
            DateRange.WEEK -> true to 7
            DateRange.MONTH -> true to 30
            DateRange.YEAR -> true to 365
        }
        // Force non-lightweight for weight to ensure model series (W_obs/W_adj_pred) are populated
        val resp = plotsApiService.getWeightPlot(simple = false, days = days)
          return resp.W_obs.map { Entry(toDaysSinceEpoch(it.time_index), it.value) }
    }

    suspend fun getMetabolismData(dateRange: DateRange): List<Entry> {
        val (simple, days) = when (dateRange) {
            DateRange.WEEK -> true to 7
            DateRange.MONTH -> true to 30
            DateRange.YEAR -> true to 365
        }
        // Force non-lightweight for metabolism to ensure model series are populated
        val resp = plotsApiService.getMetabolismPlot(simple = false, days = days)
          return resp.M_base.map { Entry(toDaysSinceEpoch(it.time_index), it.value) }
    }

    suspend fun getEnergyBalanceData(dateRange: DateRange): List<Entry> {
        val (simple, days) = when (dateRange) {
            DateRange.WEEK -> true to 7
            DateRange.MONTH -> true to 30
            DateRange.YEAR -> true to 365
        }
        // Prefer full path as well to keep consistency; backend falls back efficiently if needed
        val resp = plotsApiService.getEnergyBalancePlot(simple = false, days = days)
          return resp.calories_unnorm.map { Entry(toDaysSinceEpoch(it.time_index), it.value) }
    }
}