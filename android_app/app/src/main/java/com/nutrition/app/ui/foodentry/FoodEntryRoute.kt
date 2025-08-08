package com.nutrition.app.ui.foodentry

import androidx.compose.runtime.Composable
import androidx.hilt.navigation.compose.hiltViewModel
import com.nutrition.app.util.toLocalDateTime

@Composable
fun FoodEntryRoute(
    viewModel: FoodEntryViewModel = hiltViewModel(),
    onSave: () -> Unit,
    onCancel: () -> Unit
) {
    FoodEntryForm(
        onSave = { foodName, quantity, loggedAt ->
            viewModel.saveFoodLogEntry(
                foodName = foodName,
                quantity = quantity,
                loggedAt = loggedAt.toLocalDateTime()
            )
            onSave()
        },
        onCancel = onCancel
    )
}