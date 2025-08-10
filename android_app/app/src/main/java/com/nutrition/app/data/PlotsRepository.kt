package com.nutrition.app.data

import com.nutrition.app.data.DateRange
import com.github.mikephil.charting.data.Entry
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class PlotsRepository @Inject constructor(
    private val plotsApiService: PlotsApiService
) {
    suspend fun getWeightData(dateRange: DateRange): List<Entry> {
        val resp = plotsApiService.getWeightPlot()
        // Use observed weight series (W_obs) for the chart's main line
        return resp.W_obs.map { Entry(it.time_index.toFloat(), it.value) }
    }

    suspend fun getMetabolismData(dateRange: DateRange): List<Entry> {
        val resp = plotsApiService.getMetabolismPlot()
        return resp.M_base.map { Entry(it.time_index.toFloat(), it.value) }
    }

    suspend fun getEnergyBalanceData(dateRange: DateRange): List<Entry> {
        val resp = plotsApiService.getEnergyBalancePlot()
        // Choose calories_unnorm as primary series; could combine later if needed
        return resp.calories_unnorm.map { Entry(it.time_index.toFloat(), it.value) }
    }
}