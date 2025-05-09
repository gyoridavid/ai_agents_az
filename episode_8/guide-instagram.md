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

<img width="973" alt="Screenshot 2025-04-28 at 2 18 42 PM" src="https://github.com/user-attachments/assets/60e0b931-0b5b-4558-9559-b0668e474504" />
<img width="1536" alt="Screenshot 2025-04-28 at 2 16 45 PM" src="https://github.com/user-attachments/assets/a10295a3-1d0e-452e-978b-4cf66ce62205" />
<img width="1039" alt="Screenshot 2025-04-28 at 2 16 29 PM" src="https://github.com/user-attachments/assets/37b25c52-683b-4e33-b1f4-075a98a51979" />
<img width="1050" alt="Screenshot 2025-04-28 at 2 16 23 PM" src="https://github.com/user-attachments/assets/4aa73ed4-183d-4854-9db8-08761978b459" />
<img width="1030" alt="Screenshot 2025-04-28 at 2 15 58 PM" src="https://github.com/user-attachments/assets/cc0a4ffa-f05a-4d90-b79c-92d18b1cdec3" />
<img width="1032" alt="Screenshot 2025-04-28 at 2 15 27 PM" src="https://github.com/user-attachments/assets/7499e27e-297b-490c-b20c-9a9061516a2b" />
<img width="1038" alt="Screenshot 2025-04-28 at 2 15 14 PM" src="https://github.com/user-attachments/assets/7722d99f-5797-4840-809d-dfc400697c20" />
<img width="1723" alt="Screenshot 2025-04-28 at 2 15 07 PM" src="https://github.com/user-attachments/assets/b5fa24c0-981a-4a18-af68-c3656d2f0ec5" />


### 4. Create an access token


