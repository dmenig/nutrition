package com.nutrition.app.ui.sportentry

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import java.util.Date
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.foundation.text.KeyboardOptions

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SportEntryForm(
    modifier: Modifier = Modifier,
    activityName: String? = "",
    duration: String? = "",
    caloriesExpended: Float? = 0f,
    carriedWeight: String = "",
    distance: String = "",
    sportNames: List<String>,
    onSave: (String, String, String?, String?, String) -> Unit,
    onCancel: () -> Unit
) {
    var selectedActivityName by remember(sportNames, activityName) {
        mutableStateOf(
            activityName ?: sportNames.firstOrNull() ?: ""
        )
    }
    var durationText by remember { mutableStateOf(duration ?: "") }
    var caloriesExpendedText by remember { mutableStateOf(caloriesExpended?.toString() ?: "") }
    var carriedWeightText by remember { mutableStateOf(carriedWeight) }
    var distanceText by remember { mutableStateOf(distance) }
    var expanded by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Sport Entry") })
        },
        content = { paddingValues ->
            Column(
                modifier = modifier
                    .padding(paddingValues)
                    .padding(16.dp)
                    .fillMaxWidth()
            ) {
                ExposedDropdownMenuBox(
                    expanded = expanded,
                    onExpandedChange = { expanded = !expanded },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    OutlinedTextField(
                        value = selectedActivityName,
                        onValueChange = { selectedActivityName = it },
                        readOnly = true,
                        label = { Text("Activity Name") },
                        trailingIcon = {
                            ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
                        },
                        modifier = Modifier
                            .menuAnchor()
                            .fillMaxWidth()
                    )
                    ExposedDropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false }
                    ) {
                        sportNames.forEach { sportName ->
                            DropdownMenuItem(
                                text = { Text(sportName) },
                                onClick = {
                                    selectedActivityName = sportName
                                    expanded = false
                                }
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(16.dp))
                OutlinedTextField(
                    value = durationText,
                    onValueChange = { durationText = it },
                    label = { Text("Duration (minutes)") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(16.dp))
                OutlinedTextField(
                    value = caloriesExpendedText,
                    onValueChange = { caloriesExpendedText = it },
                    label = { Text("Calories Expended") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(16.dp))
                OutlinedTextField(
                    value = carriedWeightText,
                    onValueChange = { carriedWeightText = it },
                    label = { Text("Carried Weight (kg, optional)") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(16.dp))
                OutlinedTextField(
                    value = distanceText,
                    onValueChange = { distanceText = it },
                    label = { Text("Distance (m, optional)") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(16.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Button(onClick = onCancel) {
                        Text("Cancel")
                    }
                    Button(onClick = {
                        val safeActivityName = selectedActivityName.ifBlank { sportNames.firstOrNull() ?: "" }
                        onSave(
                            safeActivityName,
                            durationText,
                            carriedWeightText,
                            distanceText,
                            caloriesExpendedText
                        )
                    }) {
                        Text("Save")
                    }
                }
            }
        }
    )
}