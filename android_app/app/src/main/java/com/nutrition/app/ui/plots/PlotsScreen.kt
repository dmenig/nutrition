package com.nutrition.app.ui.plots

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.hilt.navigation.compose.hiltViewModel
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.formatter.ValueFormatter
import com.nutrition.app.R
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun PlotsScreen(viewModel: PlotsViewModel = hiltViewModel()) {
    var selectedTabIndex by remember { mutableStateOf(0) }
    val plotTitles = listOf("Weight", "Metabolism", "Energy Balance")
    val selectedDateRange by viewModel.selectedDateRange.collectAsState()
    var expanded by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            Column {
                TabRow(selectedTabIndex = selectedTabIndex) {
                    plotTitles.forEachIndexed { index, title ->
                        Tab(
                            selected = selectedTabIndex == index,
                            onClick = { selectedTabIndex = index },
                            text = { Text(title) }
                        )
                    }
                }
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { expanded = true }
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(text = "Date Range: ${selectedDateRange.name.replace("_", " ")}")
                    Icon(imageVector = Icons.Default.ArrowDropDown, contentDescription = "Select Date Range")
                    DropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false }
                    ) {
                        DateRange.values().forEach { dateRange ->
                            DropdownMenuItem(
                                text = { Text(dateRange.name.replace("_", " ")) },
                                onClick = {
                                    viewModel.setDateRange(dateRange)
                                    expanded = false
                                }
                            )
                        }
                    }
                }
            }
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            when (selectedTabIndex) {
                0 -> LineChartComposable(plotType = "Weight")
                1 -> LineChartComposable(plotType = "Metabolism")
                2 -> LineChartComposable(plotType = "Energy Balance")
            }
        }
    }
}

@Composable
fun LineChartComposable(plotType: String, viewModel: PlotsViewModel = hiltViewModel()) {
    val data by when (plotType) {
        "Weight" -> viewModel.weightData.collectAsState()
        "Metabolism" -> viewModel.metabolismData.collectAsState()
        "Energy Balance" -> viewModel.energyBalanceData.collectAsState()
        else -> remember { mutableStateOf(emptyList()) }
    }

    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { context ->
            LineChart(context).apply {
                setTouchEnabled(true)
                setPinchZoom(true)
                description.isEnabled = false
                setNoDataText("No data available for $plotType")

                val mv = CustomMarkerView(context, R.layout.marker_view)
                marker = mv

                xAxis.position = XAxis.XAxisPosition.BOTTOM
                xAxis.setDrawGridLines(false)
                xAxis.valueFormatter = object : ValueFormatter() {
                    private val format = SimpleDateFormat("MMM dd", Locale.getDefault())
                    override fun getFormattedValue(value: Float): String {
                        return format.format(Date(value.toLong()))
                    }
                }

                axisRight.isEnabled = false
                axisLeft.setDrawGridLines(false)
            }
        },
        update = { chart ->
            if (data.isNotEmpty()) {
                val dataSet = LineDataSet(data, plotType)
                dataSet.setDrawValues(false)
                dataSet.setDrawCircles(true)
                dataSet.setDrawCircleHole(false)
                dataSet.circleRadius = 4f
                dataSet.lineWidth = 2f

                val lineData = LineData(dataSet)
                chart.data = lineData
                chart.invalidate()
            } else {
                chart.clear()
            }
        }
    )
}