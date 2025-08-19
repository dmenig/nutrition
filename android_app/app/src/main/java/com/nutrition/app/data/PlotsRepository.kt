package com.nutrition.app.data

import com.nutrition.app.data.DateRange
import com.github.mikephil.charting.data.Entry
import javax.inject.Inject
import javax.inject.Singleton
import android.util.Log

@Singleton
class PlotsRepository @Inject constructor(
    private val plotsApiService: PlotsApiService
) {
    // MPAndroidChart uses Float for X; avoid precision loss by using days-since-epoch on the axis.
    // Normalize backend time_index which may be in ms, seconds, or days (older servers).
    private fun normalizeToEpochMs(raw: Long): Long {
        return when {
            // Clearly in milliseconds (>= ~2001-09-09 in ms). Modern servers send ms.
            raw >= 1_000_000_000_000L -> raw
            // Likely in seconds (>= ~2001-09-09 in seconds too high to be days)
            raw >= 100_000_000L -> raw * 1_000L
            // Otherwise treat as days-since-epoch
            else -> raw * 86_400_000L
        }
    }

    private fun toDaysSinceEpochFromAny(rawTimeIndex: Long): Float {
        val epochMs = normalizeToEpochMs(rawTimeIndex)
        return (epochMs / 86_400_000.0).toFloat()
    }

    // Always rely on backend-provided time_index; no client-side rebasing.

    

    suspend fun getWeightData(dateRange: DateRange): List<Entry> {
        val days = when (dateRange) {
            DateRange.WEEK -> 7
            DateRange.MONTH -> 30
            DateRange.YEAR -> 365
        }
        val resp = plotsApiService.getWeightPlot(days = days)
        // Use model predictions only; no fallback to observed weights
        val series = resp.W_adj_pred
        val entries = series.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
        if (entries.isNotEmpty()) {
            val preview = entries.take(3).joinToString { "(${it.x}, ${it.y})" }
            Log.d("PlotsRepository", "Weight entries preview: $preview")
        } else {
            Log.d("PlotsRepository", "Weight entries empty")
        }
        return entries
    }

    suspend fun getMetabolismData(dateRange: DateRange): List<Entry> {
        val days = when (dateRange) {
            DateRange.WEEK -> 7
            DateRange.MONTH -> 30
            DateRange.YEAR -> 365
        }
        val resp = plotsApiService.getMetabolismPlot(days = days)
        return resp.M_base.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
    }

    suspend fun getEnergyBalanceData(dateRange: DateRange): List<Entry> {
        val days = when (dateRange) {
            DateRange.WEEK -> 7
            DateRange.MONTH -> 30
            DateRange.YEAR -> 365
        }
        val resp = plotsApiService.getEnergyBalancePlot(days = days)
        return resp.calories_unnorm.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
    }
}