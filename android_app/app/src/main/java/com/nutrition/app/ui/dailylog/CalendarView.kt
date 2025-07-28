package com.nutrition.app.ui.dailylog

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun CalendarView(
    selectedDate: Date,
    onDateSelected: (Date) -> Unit,
    modifier: Modifier = Modifier
) {
    val calendar = remember { Calendar.getInstance() }
    var currentMonth by remember { mutableStateOf(calendar.get(Calendar.MONTH)) }
    var currentYear by remember { mutableStateOf(calendar.get(Calendar.YEAR)) }

    // Update calendar instance when month or year changes
    LaunchedEffect(currentMonth, currentYear) {
        calendar.set(Calendar.MONTH, currentMonth)
        calendar.set(Calendar.YEAR, currentYear)
        calendar.set(Calendar.DAY_OF_MONTH, 1) // Set to first day of month
    }

    Column(modifier = modifier.fillMaxWidth()) {
        // Month navigation
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = {
                if (currentMonth == Calendar.JANUARY) {
                    currentMonth = Calendar.DECEMBER
                    currentYear--
                } else {
                    currentMonth--
                }
            }) {
                Icon(Icons.Default.ArrowBack, contentDescription = "Previous Month")
            }
            Text(
                text = SimpleDateFormat("MMMM yyyy", Locale.getDefault()).format(calendar.time),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            IconButton(onClick = {
                if (currentMonth == Calendar.DECEMBER) {
                    currentMonth = Calendar.JANUARY
                    currentYear++
                } else {
                    currentMonth++
                }
            }) {
                Icon(Icons.Default.ArrowForward, contentDescription = "Next Month")
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Weekday headers
        Row(modifier = Modifier.fillMaxWidth()) {
            val weekdays = listOf("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
            weekdays.forEach { day ->
                Text(
                    text = day,
                    modifier = Modifier.weight(1f),
                    textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Spacer(modifier = Modifier.height(4.dp))

        // Days of the month
        val daysInMonth = remember(currentMonth, currentYear) {
            calendar.getActualMaximum(Calendar.DAY_OF_MONTH)
        }
        val firstDayOfWeek = remember(currentMonth, currentYear) {
            calendar.get(Calendar.DAY_OF_WEEK) - 1 // 0 for Sunday, 1 for Monday, etc.
        }

        LazyVerticalGrid(
            columns = GridCells.Fixed(7),
            modifier = Modifier.fillMaxWidth(),
            userScrollEnabled = false // Disable scrolling for the grid itself
        ) {
            // Empty cells for days before the 1st of the month
            items(firstDayOfWeek) {
                Spacer(modifier = Modifier.size(40.dp))
            }

            // Days of the month
            items((1..daysInMonth).toList()) { dayOfMonth ->
                val dayDate = remember(dayOfMonth, currentMonth, currentYear) {
                    val dayCalendar = Calendar.getInstance().apply {
                        set(Calendar.YEAR, currentYear)
                        set(Calendar.MONTH, currentMonth)
                        set(Calendar.DAY_OF_MONTH, dayOfMonth)
                        set(Calendar.HOUR_OF_DAY, 0)
                        set(Calendar.MINUTE, 0)
                        set(Calendar.SECOND, 0)
                        set(Calendar.MILLISECOND, 0)
                    }
                    dayCalendar.time
                }

                val isSelected = remember(selectedDate, dayDate) {
                    val selectedCal = Calendar.getInstance().apply { time = selectedDate }
                    val dayCal = Calendar.getInstance().apply { time = dayDate }
                    selectedCal.get(Calendar.YEAR) == dayCal.get(Calendar.YEAR) &&
                            selectedCal.get(Calendar.MONTH) == dayCal.get(Calendar.MONTH) &&
                            selectedCal.get(Calendar.DAY_OF_MONTH) == dayCal.get(Calendar.DAY_OF_MONTH)
                }

                val isToday = remember(dayDate) {
                    val todayCal = Calendar.getInstance()
                    val dayCal = Calendar.getInstance().apply { time = dayDate }
                    todayCal.get(Calendar.YEAR) == dayCal.get(Calendar.YEAR) &&
                            todayCal.get(Calendar.MONTH) == dayCal.get(Calendar.MONTH) &&
                            todayCal.get(Calendar.DAY_OF_MONTH) == dayCal.get(Calendar.DAY_OF_MONTH)
                }

                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(
                            when {
                                isSelected -> MaterialTheme.colorScheme.primary
                                isToday -> MaterialTheme.colorScheme.secondary.copy(alpha = 0.3f)
                                else -> Color.Transparent
                            }
                        )
                        .clickable { onDateSelected(dayDate) },
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = dayOfMonth.toString(),
                        color = if (isSelected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = if (isToday || isSelected) FontWeight.Bold else FontWeight.Normal
                    )
                }
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun CalendarViewPreview() {
    CalendarView(selectedDate = Date(), onDateSelected = {})
}