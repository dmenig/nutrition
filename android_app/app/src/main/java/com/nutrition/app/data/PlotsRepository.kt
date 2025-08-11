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

    // Some servers may return ordinal indices starting at 0 for recent series; rebase to today.
    private fun maybeRebaseOrdinalToToday(xValues: List<Float>): List<Float> {
        if (xValues.isEmpty()) return xValues
        val maxX = xValues.maxOrNull() ?: return xValues
        val minX = xValues.minOrNull() ?: return xValues
        // If values are a small range like [0..400], treat as ordinal days
        if (minX >= 0f && maxX <= 4000f) {
            val todayDays = (System.currentTimeMillis() / 86_400_000.0).toFloat()
            val shift = todayDays - maxX
            return xValues.map { it + shift }
        }
        return xValues
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
        // Use lightweight for reliability under slow network/server
        val resp = plotsApiService.getWeightPlot(simple = true, days = days)
        var entries = resp.W_obs.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
        if (entries.isEmpty()) {
            // Fallback to adjusted predicted weight if observed is empty
            try {
                val full = plotsApiService.getWeightPlot(simple = false, days = days)
                entries = full.W_adj_pred.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
            } catch (_: Exception) { }
        }
        // Rebase if backend sent ordinal indices
        val rebasedX = maybeRebaseOrdinalToToday(entries.map { it.x })
        val rebased = entries.indices.map { i -> Entry(rebasedX[i], entries[i].y) }
        return dropIfConstant(rebased) { it.y }
    }

    suspend fun getMetabolismData(dateRange: DateRange): List<Entry> {
        val (simple, days) = when (dateRange) {
            DateRange.WEEK -> true to 7
            DateRange.MONTH -> true to 30
            DateRange.YEAR -> true to 365
        }
        // Force non-lightweight for metabolism to ensure model series are populated
        val resp = plotsApiService.getMetabolismPlot(simple = true, days = days)
        val raw = resp.M_base.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
        val rebasedX = maybeRebaseOrdinalToToday(raw.map { it.x })
        val entries = raw.indices.map { i -> Entry(rebasedX[i], raw[i].y) }
        return dropIfConstant(entries) { it.y }
    }

    suspend fun getEnergyBalanceData(dateRange: DateRange): List<Entry> {
        val (simple, days) = when (dateRange) {
            DateRange.WEEK -> true to 7
            DateRange.MONTH -> true to 30
            DateRange.YEAR -> true to 365
        }
        // Prefer full path as well to keep consistency; backend falls back efficiently if needed
        val resp = plotsApiService.getEnergyBalancePlot(simple = true, days = days)
        val raw = resp.calories_unnorm.map { Entry(toDaysSinceEpochFromAny(it.time_index), it.value) }
        val rebasedX = maybeRebaseOrdinalToToday(raw.map { it.x })
        return raw.indices.map { i -> Entry(rebasedX[i], raw[i].y) }
    }
}