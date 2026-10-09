import zipfile
import os
import time

zip_path = "FAB_FL_Client_Source.zip"
temp_zip = "FAB_FL_Client_Source_temp.zip"

files_to_update = [
    "Client.py",
    "FAB-FL-Software/client_app.py"
]

print(f"Updating {zip_path} with {files_to_update}...")
start_time = time.time()

# We recreate the zip file to avoid duplicate entries and save space
with zipfile.ZipFile(zip_path, 'r') as zin, zipfile.ZipFile(temp_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        # Keep items that are not being updated
        if item.filename not in files_to_update:
            zout.writestr(item, zin.read(item.filename))
            
    # Now add the updated/new files
    for f in files_to_update:
        if os.path.exists(f):
            print(f"Adding updated file: {f}")
            zout.write(f, arcname=f)
        else:
            print(f"Warning: {f} not found!")

print("Replacing original zip file...")
os.replace(temp_zip, zip_path)
print(f"Done in {time.time() - start_time:.2f} seconds!")
