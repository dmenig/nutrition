package com.nutrition.app.util

import okhttp3.Interceptor
import okhttp3.Response

class ErrorReportingInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        return try {
            val response = chain.proceed(chain.request())
            if (!response.isSuccessful) {
                val code = response.code
                ErrorReporter.show("Network error (${code}) while calling ${chain.request().url.encodedPath}")
            }
            response
        } catch (e: Exception) {
            ErrorReporter.show("Network failure: ${e.message ?: e::class.java.simpleName}")
            throw e
        }
    }
}


