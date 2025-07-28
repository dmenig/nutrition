package com.nutrition.app.ui.customfood

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.nutrition.app.data.model.CustomFood

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CustomFoodListScreen(
    onAddFoodClick: () -> Unit,
    viewModel: CustomFoodViewModel = hiltViewModel()
) {
    val customFoods by viewModel.customFoods.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Custom Foods") })
        },
        floatingActionButton = {
            FloatingActionButton(onClick = onAddFoodClick) {
                Icon(Icons.Filled.Add, "Add new custom food")
            }
        }
    ) { paddingValues ->
        if (customFoods.isEmpty()) {
            Text(
                text = "No custom foods added yet. Click the '+' button to add one.",
                modifier = Modifier
                    .padding(paddingValues)
                    .fillMaxSize()
                    .wrapContentSize()
            )
        } else {
            LazyColumn(
                modifier = Modifier
                    .padding(paddingValues)
                    .fillMaxSize()
            ) {
                items(customFoods) { food ->
                    CustomFoodItem(food = food, onDeleteClick = { viewModel.deleteCustomFood(it) })
                }
            }
        }
    }
}

@Composable
fun CustomFoodItem(food: CustomFood, onDeleteClick: (CustomFood) -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(text = food.name, style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(4.dp))
                Text(text = "Calories: ${food.caloriesPer100g}kcal", style = MaterialTheme.typography.bodySmall)
                Text(text = "Protein: ${food.proteinPer100g}g", style = MaterialTheme.typography.bodySmall)
                Text(text = "Carbs: ${food.carbohydratesPer100g}g", style = MaterialTheme.typography.bodySmall)
                Text(text = "Fat: ${food.fatPer100g}g", style = MaterialTheme.typography.bodySmall)
            }
            IconButton(onClick = { onDeleteClick(food) }) {
                Icon(Icons.Filled.Delete, "Delete custom food")
            }
        }
    }
}