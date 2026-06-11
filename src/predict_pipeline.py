import subprocess
import sys
import os

# PATHS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GP_MAP = {
    "australia": "australian_gp",
    "china": "chinese_gp",
    "miami": "miami_gp",
    "japan": "japanese_gp",
    "canada": "canada_gp"
}

def run_script(script_path):
    print(f"🚀 Running: {script_path}")
    result = subprocess.run(["python3", script_path], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Error in {script_path}")
        sys.exit(1)
    print(f"✅ Completed: {script_path}\n")

def run_pipeline(gp_input):
    if gp_input not in GP_MAP:
        print(f"❌ Unknown GP: {gp_input}")
        return False

    gp_folder = GP_MAP[gp_input]
    gp_path = os.path.join(BASE_DIR, gp_folder)

    print(f"\n🌟 STARTING PIPELINE FOR: {gp_input.upper()} GP\n" + "="*40)

    # 1. Collect Data
    run_script(os.path.join(gp_path, "src/data_collection/collect_data.py"))

    # 2. Feature Engineering
    run_script(os.path.join(gp_path, "src/features/feature_engineering.py"))

    # 3. Merge Dataset (Global)
    run_script(os.path.join(BASE_DIR, "src/merge_dataset.py"))

    # 4. Train Model
    run_script(os.path.join(gp_path, "src/models/train_model.py"))

    # 5. Predict Race
    run_script(os.path.join(gp_path, "src/models/predict_race.py"))

    print(f"🏁 Execution finished for {gp_input.upper()} GP!\n")
    return True

def main():
    # Check if user wants to run all from schedule.txt
    if len(sys.argv) > 1 and sys.argv[1].lower() == "all":
        schedule_path = os.path.join(BASE_DIR, "schedule.txt")
        if not os.path.exists(schedule_path):
            print(f"❌ schedule.txt not found at {schedule_path}")
            sys.exit(1)
        
        with open(schedule_path, "r") as f:
            # Read names, remove trailing punctuation (like Canada.) and whitespace
            races = [line.strip().lower().rstrip(".") for line in f if line.strip()]
            
        print(f"📅 Schedule detected: {', '.join(races)}")
        for race in races:
            success = run_pipeline(race)
            if not success:
                sys.exit(1)
        
        # After all runs, show final summary
        run_script(os.path.join(BASE_DIR, "src/evaluate_performance.py"))
        print("\n🏆 全て (All) predictions completed in scheduled order!")
        return

    if len(sys.argv) < 2:
        print("Usage: python3 src/predict_pipeline.py <gp_name> | all")
        print(f"Supported GPs: {', '.join(GP_MAP.keys())}")
        sys.exit(1)

    gp_input = sys.argv[1].lower().replace("_gp", "")
    run_pipeline(gp_input)

if __name__ == "__main__":
    main()
