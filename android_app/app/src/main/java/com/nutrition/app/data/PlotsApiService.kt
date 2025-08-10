package com.nutrition.app.data

import retrofit2.http.GET

// Data models matching backend schemas
data class PlotPoint(
    val time_index: Int,
    val value: Float
)

data class WeightPlotResponse(
    val W_obs: List<PlotPoint>,
    val W_adj_pred: List<PlotPoint>
)

data class MetabolismPlotResponse(
    val M_base: List<PlotPoint>
)

data class EnergyBalancePlotResponse(
    val calories_unnorm: List<PlotPoint>,
    val C_exp_t: List<PlotPoint>
)

interface PlotsApiService {
    @GET("/api/v1/plots/weight")
    suspend fun getWeightPlot(): WeightPlotResponse

    @GET("/api/v1/plots/metabolism")
    suspend fun getMetabolismPlot(): MetabolismPlotResponse

    @GET("/api/v1/plots/energy-balance")
    suspend fun getEnergyBalancePlot(): EnergyBalancePlotResponse
}