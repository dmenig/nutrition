package com.nutrition.app.data

import com.nutrition.app.data.DateRange
import com.github.mikephil.charting.data.Entry
import javax.inject.Inject
import javax.inject.Singleton

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

    private fun <T> dropIfConstant(entries: List<T>, valueOf: (T) -> Float): List<T> {
        if (entries.isEmpty()) return entries
        val first = valueOf(entries.first())
        val epsilon = 1e-6f
        val allSame = entries.all { kotlin.math.abs(valueOf(it) - first) < epsilon }
        return if (allSame) emptyList() else entries
    }

    suspend fun getWeightData(dateRange: DateRange): List<Entry> {
        val (simple, days) = when (dateRange) {
            DateRange.WEEK -> true to 7
            DateRange.MONTH -> true to 30
            DateRange.YEAR -> true to 365
        }
        // Force non-lightweight for weight to ensure model series (W_obs/W_adj_pred) are populated
        val resp = plotsApiService.getWeightPlot(simple = false, days = days)
        val entries = resp.W_obs.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
        return dropIfConstant(entries) { it.y }
    }

    suspend fun getMetabolismData(dateRange: DateRange): List<Entry> {
        val (simple, days) = when (dateRange) {
            DateRange.WEEK -> true to 7
            DateRange.MONTH -> true to 30
            DateRange.YEAR -> true to 365
        }
        // Force non-lightweight for metabolism to ensure model series are populated
        val resp = plotsApiService.getMetabolismPlot(simple = false, days = days)
        val entries = resp.M_base.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
        return dropIfConstant(entries) { it.y }
    }

    suspend fun getEnergyBalanceData(dateRange: DateRange): List<Entry> {
        val (simple, days) = when (dateRange) {
            DateRange.WEEK -> true to 7
            DateRange.MONTH -> true to 30
            DateRange.YEAR -> true to 365
        }
        // Prefer full path as well to keep consistency; backend falls back efficiently if needed
        val resp = plotsApiService.getEnergyBalancePlot(simple = false, days = days)
        return resp.calories_unnorm.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
    }
}