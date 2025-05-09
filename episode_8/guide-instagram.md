# Setting up Instagram for automations with n8n

## Convert the Instagram account to a business account

> [!IMPORTANT]  
> Converting the Instagram account to a business account will make the account public

In the Instagram application, follow these steps to convert your account to a business account.

<table>
  <tr>
    <td>
      1. Open the settings, and select `Account type and tools`
      <img src="https://github.com/user-attachments/assets/adffda31-f8ce-4a49-a971-40e5cd5f5395" alt="" />
    </td>
    <td>
      2. Click on `Switch to professinal account`
      <img src="https://github.com/user-attachments/assets/5ea13aeb-b6ac-40b4-906f-2fc563625c52" alt="" />
    </td>
    <td>
      3. Click on `Next`
      <img src="https://github.com/user-attachments/assets/10942cc8-6045-4ca1-a14a-241a2d94591c" />
    </td>
  </tr>
  <tr>
    <td>
      4. Select the type of account your want to create
      <img src="https://github.com/user-attachments/assets/c9e19bf8-b48d-49e7-b560-e8edf423233d" alt="" />
    </td>
    <td>
      5. Select the right category, that describe your account
      <img src="https://github.com/user-attachments/assets/eb4680b2-1d38-476c-8fb0-c428d1e12d04" alt="" />
    </td>
    <td>
      6. Confirm the choice
      <img src="https://github.com/user-attachments/assets/22ee13cf-f751-4cea-9fc0-be466c885b02" alt="" />
    </td>
  </tr>
</table>

## Create credentials to use with n8n

### 1. Create a Facebook page

Navigate to [https://www.facebook.com/pages/create](https://www.facebook.com/pages/create) to create a new Facebook page.
Fill out the required parameters (name, category) and create the page.

<img width="1554" alt="Screenshot 2025-05-05 at 10 02 00 AM" src="https://github.com/user-attachments/assets/f0e321a2-7b85-4b23-b85b-f8e63281b6b9" />

Now, you need to be acting on behalf of the page - if it's not active for some reason, select it from the menu.

<img width="372" alt="image" src="https://github.com/user-attachments/assets/7aaaccc8-4b0a-4f72-a8c9-f2ecc453584a" />

### 2. Connect Instagram account to the Facebook page

Click on your page's name on the left side.

<img width="379" alt="image" src="https://github.com/user-attachments/assets/fd1ed1d8-8890-4636-a4ae-0496db252283" />

Select `Settings` from the menu.

<img width="368" alt="image" src="https://github.com/user-attachments/assets/641904ec-764e-4d56-bede-618da33db7fe" />

Scroll down to `Permissions` and select `Linked accounts`.

<img width="369" alt="image" src="https://github.com/user-attachments/assets/a024ad7e-8358-4329-a864-c7b621810405" />

Select `Instagram`.

<img width="740" alt="image" src="https://github.com/user-attachments/assets/141164a0-c88e-46a0-a335-466f81297bc6" />

Click on `Connect account` and sign in with your Instagram account.

<img width="708" alt="image" src="https://github.com/user-attachments/assets/5324b945-621b-4f2b-b513-9b51e5b5c071" />

Click on `Connect`.

<img width="573" alt="image" src="https://github.com/user-attachments/assets/85cad094-a815-441a-a0e5-84081fd0ee2d" />

Click on `Confirm`.

<img width="570" alt="image" src="https://github.com/user-attachments/assets/df5bb377-22a8-41de-b99d-0f80052cb372" />

Click on `Continue`.

<img width="561" alt="image" src="https://github.com/user-attachments/assets/81fd3143-f00b-4e33-9521-12c0c9ce623d" />

You are done.

<img width="713" alt="image" src="https://github.com/user-attachments/assets/ce4c0bbe-beed-4c6c-8731-23216efbccb5" />

### 3. Create a Facebook application

Head to [https://developers.facebook.com/apps](https://developers.facebook.com/apps) and click on `Create app`

<img width="1027" alt="image" src="https://github.com/user-attachments/assets/ca5c79b9-974a-4c55-8f88-6def8c1b98d1" />

Select the `Other` use case and hit `Next`.

<img width="1033" alt="image" src="https://github.com/user-attachments/assets/31f296bd-3b47-432b-9037-7974e821a311" />

Select `Business` app type.

<img width="818" alt="image" src="https://github.com/user-attachments/assets/2ab06dd1-33ba-4d87-8365-3637c8327877" />

Review, and click on `Create app`

<img width="815" alt="image" src="https://github.com/user-attachments/assets/c8cc8e1d-c006-428a-9bfa-8cc78a2932c9" />

Add the Instagram product to your app.

<img width="1008" alt="image" src="https://github.com/user-attachments/assets/4186f90b-b619-4c1c-9ea8-bfc5294eaeb4" />

You don't need to configure it, just leave it as it is. 


### 4. Create an access token

From the tools menu, select the `Graph API Explorer`

<img width="877" alt="image" src="https://github.com/user-attachments/assets/006a40a7-894a-4bd0-9483-0d495b77206f" />

You'll see this interface below. <img width="541" alt="image" src="https://github.com/user-attachments/assets/414ece47-2f1d-4d08-9998-ada85ffb065f" />

1. Make sure the Meta app is the one you just created
2. In the permissions panel, add the `instagram_basic` and `instagram_content_publish`
3. In the `User of Page` section, you need to select `Get Page Access Token`

<img width="533" alt="image" src="https://github.com/user-attachments/assets/51433f7a-a1a0-40e8-971a-be1f7064ebd6" />

You will be prompted to authenticate with your Facebook account. In the process, select the previously created Facebook page.

<img width="566" alt="Screenshot 2025-05-09 at 10 03 39 AM" src="https://github.com/user-attachments/assets/edffac2a-5e25-4f95-94d2-90b7270de8ad" />

<img width="567" alt="Screenshot 2025-05-09 at 10 03 59 AM" src="https://github.com/user-attachments/assets/e2a613c8-21aa-45ce-9757-7aead8808759" />



Now, this is important: once you are done, you need to click on the blue `Generate Access Token` and re-authenticate with your Facebook account.

This time, you'll have an extra step included in the process, you can select the Instagram account that is connected to the Facebook page you've selected.

<img width="568" alt="Screenshot 2025-05-09 at 10 06 02 AM" src="https://github.com/user-attachments/assets/576cf8ed-3e9f-44bf-a966-1e1c03f133db" />


Finish this step, and copy the access token.

Head to the [Facebook token debugger tool](https://developers.facebook.com/tools/debug/accesstoken) and paste in the access token, and click `debug`.

