import os
import re
import shutil

def migrate_lr_dirs(models_dir=".models"):
    """
    Renames directories in .models/<group>/<run_name> from lrXe-Y to lrXe-0Y
    to match the new naming standard (unified with Python's :g format).
    """
    if not os.path.exists(models_dir):
        print(f"Directory {models_dir} not found.")
        return

    # Pattern: 'lr' followed by digits/dots, then 'e[+-]', then a single digit, then '_' or end of string
    # We want to insert a '0' before that single digit.
    regex = re.compile(r"^(lr[\d\.]+)e([+-])([0-9])(_|$)")

    renamed_count = 0
    
    # Iterate through group directories
    for group in os.listdir(models_dir):
        group_path = os.path.join(models_dir, group)
        if not os.path.isdir(group_path):
            continue
            
        # Iterate through run directories
        for run_name in os.listdir(group_path):
            run_path = os.path.join(group_path, run_name)
            if not os.path.isdir(run_path):
                continue
            
            match = regex.match(run_name)
            if match:
                prefix = match.group(1)
                sign = match.group(2)
                exponent = match.group(3)
                suffix = match.group(4)
                
                # We want result like lr2e-05 or lr2e+04
                new_run_name = f"{prefix}e{sign}0{exponent}{suffix}" + run_name[match.end():]
                
                new_run_path = os.path.join(group_path, new_run_name)
                
                # Check for collisions (though unlikely)
                if os.path.exists(new_run_path):
                    print(f"⚠️  Skipping: '{run_path}' -> '{new_run_name}' already exists.")
                    continue
                
                print(f"✅ Renaming: '{run_name}' -> '{new_run_name}'")
                os.rename(run_path, new_run_path)
                renamed_count += 1

    print(f"\nMigration complete. Renamed {renamed_count} directories.")

if __name__ == "__main__":
    # Check if we are in the root directory by looking for .models or dllm/
    if not os.path.exists(".models") and os.path.exists("../.models"):
        os.chdir("..")
        
    migrate_lr_dirs()
