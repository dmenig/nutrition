package com.nutrition.app.util

import android.content.Context
import android.content.SharedPreferences

class AuthTokenStore(context: Context) {
    private val prefs: SharedPreferences =
        context.getSharedPreferences("auth_prefs", Context.MODE_PRIVATE)

    fun getToken(): String? = prefs.getString(KEY_TOKEN, null)

    fun setToken(token: String?) {
        prefs.edit().putString(KEY_TOKEN, token).apply()
    }

    companion object {
        private const val KEY_TOKEN = "access_token"
    }
}


