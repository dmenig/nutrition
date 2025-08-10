package com.nutrition.app.util

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow

object ErrorReporter {
    private val _events: MutableSharedFlow<String> = MutableSharedFlow(replay = 0, extraBufferCapacity = 64)
    val events: SharedFlow<String> = _events

    fun show(message: String) {
        _events.tryEmit(message)
    }
}


