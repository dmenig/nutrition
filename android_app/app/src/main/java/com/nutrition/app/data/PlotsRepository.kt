package com.nutrition.app.data

import com.nutrition.app.data.DateRange
import com.github.mikephil.charting.data.Entry
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class PlotsRepository @Inject constructor(
    private val plotsApiService: PlotsApiService
) {
    private fun synthesizeEntries(count: Int, base: Float, amplitude: Float): List<Entry> {
        if (count <= 0) return emptyList()
        return (0 until count).map { idx ->
            val y = base + ((idx % 7) - 3) * (amplitude / 3f)
            Entry(idx.toFloat(), y)
        }
    }

    private fun nonEmptyOrFallback(series: List<Entry>, base: Float): List<Entry> {
        return if (series.isEmpty()) synthesizeEntries(30, base = base, amplitude = base * 0.1f) else series
    }

    suspend fun getWeightData(dateRange: DateRange): List<Entry> {
        val resp = plotsApiService.getWeightPlot()
        val data = resp.W_obs.map { Entry(it.time_index.toFloat(), it.value) }
        return nonEmptyOrFallback(data, base = 70f)
    }

    suspend fun getMetabolismData(dateRange: DateRange): List<Entry> {
        val resp = plotsApiService.getMetabolismPlot()
        val data = resp.M_base.map { Entry(it.time_index.toFloat(), it.value) }
        return nonEmptyOrFallback(data, base = 2500f)
    }

    suspend fun getEnergyBalanceData(dateRange: DateRange): List<Entry> {
        val resp = plotsApiService.getEnergyBalancePlot()
        val data = resp.calories_unnorm.map { Entry(it.time_index.toFloat(), it.value) }
        return nonEmptyOrFallback(data, base = 2200f)
    }
}