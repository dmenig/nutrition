package com.nutrition.app.ui.customfood

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import kotlin.math.abs

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CustomFoodEntryScreen(
    onBackClick: () -> Unit,
    viewModel: CustomFoodViewModel = hiltViewModel()
) {
    var name by remember { mutableStateOf("") }
    var calories by remember { mutableStateOf("") }
    var protein by remember { mutableStateOf("") }
    var carbohydrates by remember { mutableStateOf("") }
    var fat by remember { mutableStateOf("") }
    var sugar by remember { mutableStateOf("") }
    var sfat by remember { mutableStateOf("") }
    var freeSugar by remember { mutableStateOf("") }
    var fibres by remember { mutableStateOf("") }
    var sel by remember { mutableStateOf("") }
    var alcool by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Add Custom Food") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Filled.ArrowBack, "Back")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .padding(paddingValues)
                .padding(16.dp)
                .fillMaxSize()
        ) {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                label = { Text("Food Name") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = calories,
                onValueChange = { calories = it },
                label = { Text("Calories (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = protein,
                onValueChange = { protein = it },
                label = { Text("Protein (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = carbohydrates,
                onValueChange = { carbohydrates = it },
                label = { Text("Carbohydrates (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = fat,
                onValueChange = { fat = it },
                label = { Text("Fat (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = sfat,
                onValueChange = { sfat = it },
                label = { Text("Saturated fat (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = sugar,
                onValueChange = { sugar = it },
                label = { Text("Sugar (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = freeSugar,
                onValueChange = { freeSugar = it },
                label = { Text("Free sugar (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = fibres,
                onValueChange = { fibres = it },
                label = { Text("Fibres (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = sel,
                onValueChange = { sel = it },
                label = { Text("Salt (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = alcool,
                onValueChange = { alcool = it },
                label = { Text("Alcohol (per 100g)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(16.dp))
            Button(
                onClick = {
                    val cals = calories.toFloatOrNull() ?: 0f
                    val prot = protein.toFloatOrNull()
                    val carbsVal = carbohydrates.toFloatOrNull()
                    val fatVal = fat.toFloatOrNull()
                    viewModel.createFood(
                        name = name,
                        calories = cals,
                        protein = prot,
                        carbs = carbsVal,
                        fat = fatVal,
                        sugar = sugar.toFloatOrNull(),
                        sfat = sfat.toFloatOrNull(),
                        freeSugar = freeSugar.toFloatOrNull(),
                        fibres = fibres.toFloatOrNull(),
                        sel = sel.toFloatOrNull(),
                        alcool = alcool.toFloatOrNull(),
                    ) { onBackClick() }
                },
                modifier = Modifier.fillMaxWidth()
            ) { Text("Add Food to Database") }
        }
    }
}