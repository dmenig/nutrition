package com.nutrition.app.ui.sportentry

import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.hilt.navigation.compose.hiltViewModel
import java.time.LocalDateTime

@Composable
fun SportEntryRoute(
    viewModel: SportEntryViewModel = hiltViewModel(),
    activityName: String?,
    duration: String?,
    caloriesExpended: Float?,
    onSave: () -> Unit,
    onCancel: () -> Unit
) {
    SportEntryForm(
        activityName = activityName,
        duration = duration,
        caloriesExpended = caloriesExpended ?: 0f,
        sportNames = viewModel.sportNames.collectAsState(initial = emptyList()).value,
        onSave = { formActivityName, formDuration, formCarriedWeight, formDistance, formCaloriesExpendedString ->
            viewModel.saveSportActivityEntry(
                activityName = formActivityName,
                durationMinutes = formDuration.toIntOrNull() ?: 0,
                caloriesExpended = formCaloriesExpendedString.toFloatOrNull() ?: 0f,
                carriedWeightKg = formCarriedWeight?.toFloatOrNull(),
                distanceM = formDistance?.toFloatOrNull(),
                loggedAt = LocalDateTime.now()
            )
            onSave()
        },
        onCancel = onCancel
    )
}