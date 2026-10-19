package com.chat

import android.Manifest
import android.app.Activity
import android.content.ContentResolver
import android.content.Intent
import android.content.pm.PackageManager
import android.database.Cursor
import android.net.Uri
import android.provider.ContactsContract
import android.telephony.PhoneNumberUtils
import android.util.Log
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.*

class ContactModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "ContactModule"

    @ReactMethod
    fun getContacts(promise: Promise) {
        try {
            val resolver: ContentResolver = reactApplicationContext.contentResolver
            val cursor: Cursor? = resolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                null, null, null, null
            )

            val contactList = Arguments.createArray()

            cursor?.use {
                val nameIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
                val numberIndex = it.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)

                while (it.moveToNext()) {
                    val contact = Arguments.createMap()
                    contact.putString("name", it.getString(nameIndex))
                    contact.putString("number", it.getString(numberIndex))
                    contactList.pushMap(contact)
                }
            }

            promise.resolve(contactList)
        } catch (e: Exception) {
            promise.reject("CONTACT_ERROR", "Failed to fetch contacts", e)
        }
    }

    @ReactMethod
    fun callNumber(number: String) {
        try {
            val activity: Activity? = currentActivity
            if (activity != null) {
                val intent = Intent(Intent.ACTION_CALL)
                intent.data = Uri.parse("tel:$number")
                if (ActivityCompat.checkSelfPermission(activity, Manifest.permission.CALL_PHONE) != PackageManager.PERMISSION_GRANTED) {
                    Log.e("ContactModule", "CALL_PHONE permission not granted")
                    return
                }
                activity.startActivity(intent)
            }
        } catch (e: Exception) {
            Log.e("ContactModule", "Failed to call number", e)
        }
    }
}


