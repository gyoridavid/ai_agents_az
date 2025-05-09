# Guide to create and setup the Supabase account

Start by heading to [Supabase](https://supabase.com/) and create a new free account. 

## 1. Creating the database tables

Once the account is ready, head to the `Table Editor`.

<img width="205" alt="Screenshot 2025-05-09 at 8 49 28 AM" src="https://github.com/user-attachments/assets/03f22c7c-18d5-4de0-81e0-f4fcee127647" />

To create a new database table, click on the `New table` button.

<img width="262" alt="Screenshot 2025-05-09 at 8 49 35 AM" src="https://github.com/user-attachments/assets/2a8e59f5-d20c-470e-abc8-c15966f438c1" />

### Create the `plans` table

The plans table should have 4 fields in total

1. `id` (int8) primary, just use the default settings
2. `period_type` (text) - we are using this field to index the different periods, the possible values are: `quarter` `month` and `week`
3. `period_index` (text) - this will store the value of the `period_type` - for example `2025-Q2` for the second quarter of 2025
4. `plan_value` (text) - the actual value of the plan for the given period

<img width="664" alt="Screenshot 2025-05-09 at 8 43 17 AM" src="https://github.com/user-attachments/assets/d308dd34-8fde-4cd4-8347-d2e97db1d648" />

Once ready, hit save.

### Create the `posts` table

1. `id` (int8) primary, just use the default settings
2. `created_at` (timestamp) default value `now()`
3. `post_summary` (text) - we'll store the summary of each posts for consistent posts
4. `image_path` (text) - the path of the generated image we uploaded to the Supabase storage bucket

<img width="663" alt="Screenshot 2025-05-09 at 8 43 33 AM" src="https://github.com/user-attachments/assets/917d2556-eb3d-48a1-9095-65332882fd0c" />

## Creating the storage bucket

Click on `Storage` from the menu.

<img width="200" alt="Screenshot 2025-05-09 at 9 00 15 AM" src="https://github.com/user-attachments/assets/e7cbd532-b7bd-4404-a169-2965932be24e" />

Click on `New bucket`.

<img width="242" alt="Screenshot 2025-05-09 at 9 00 21 AM" src="https://github.com/user-attachments/assets/9deca74e-1798-4733-8593-0b5593857ab4" />

Name your bucket `instagram-images` or anything that fits the naming criteria - but remember the name as you'll need to set it in n8n.

<img width="522" alt="Screenshot 2025-05-09 at 9 01 11 AM" src="https://github.com/user-attachments/assets/10cf6666-63bb-4bdc-abcc-2a0fad7e27da" />

Now, you can upload the reference image to the bucket by clicking on the `Upload files` button.

<img width="831" alt="Screenshot 2025-05-09 at 9 02 07 AM" src="https://github.com/user-attachments/assets/6cc83262-b92c-4f74-86c3-a45c9c0456f8" />

This is it, your Supabase account is ready to be used.



