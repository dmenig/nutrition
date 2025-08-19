package com.nutrition.app.util

import okhttp3.Interceptor
import okhttp3.Response

class ErrorReportingInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        return try {
            val response = chain.proceed(chain.request())
            if (!response.isSuccessful) {
                val code = response.code
                val path = chain.request().url.encodedPath
                // Suppress expected auth bootstrap errors:
                // - 400 on register when the user already exists
                // - 401 on token when login-first fails before register+retry
                val isExpectedAuthBootstrapError =
                    (code == 400 && path.endsWith("/api/v1/auth/register")) ||
                    (code == 401 && path.endsWith("/api/v1/auth/token"))
                if (!isExpectedAuthBootstrapError) {
                    ErrorReporter.show("Network error (${code}) while calling ${path}")
                }
            }
            response
        } catch (e: Exception) {
            ErrorReporter.show("Network failure: ${e.message ?: e::class.java.simpleName}")
            throw e
        }
    }
}


