import resend

resend.api_key = "re_VLcKpowV_2Z1F4oDs7WRJKZLxVixi25gU"

params = {
    "from": "onboarding@resend.dev",
    "to": ["joyallen693@gmail.com"],
    "subject": "SMTP Test",
    "html": "<strong>Hello! This is a test email from Smart Attendance System.</strong>"
}

email = resend.Emails.send(params)

print(email)