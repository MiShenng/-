import os
import shutil
import zipfile
from pathlib import Path

def create_deliverables_package():
    base_dir = Path(__file__).resolve().parent.parent.parent
    deliv_dir = base_dir / 'deliverables'
    
    if deliv_dir.exists():
        shutil.rmtree(str(deliv_dir))
    deliv_dir.mkdir(parents=True, exist_ok=True)
    
    print("Collecting files into deliverables directory...")
    
    # 1. Collect Figures
    fig_src = base_dir / 'figures'
    fig_dst = deliv_dir / 'figures'
    if fig_src.exists():
        shutil.copytree(str(fig_src), str(fig_dst))
        print(f"Copied {len(list(fig_src.iterdir()))} figures.")
        
    # 2. Collect Results (including precomputed caches so generate_all_figures works immediately)
    res_src = base_dir / 'results'
    res_dst = deliv_dir / 'results'
    if res_src.exists():
        shutil.copytree(str(res_src), str(res_dst))
        print(f"Copied results data and precomputed caches.")
        
    # 3. Collect Scripts
    script_src = base_dir / 'scripts'
    script_dst = deliv_dir / 'scripts'
    if script_src.exists():
        shutil.copytree(str(script_src), str(script_dst), ignore=shutil.ignore_patterns('__pycache__'))
        print("Copied all source code scripts.")
        
    # 3.5 Collect Data
    data_src = base_dir / 'data'
    data_dst = deliv_dir / 'data'
    if data_src.exists():
        shutil.copytree(str(data_src), str(data_dst))
        print("Copied raw data directory.")
        
    # 4. Copy main driver and metadata
    for f in ['main.py', 'requirements.txt']:
        f_path = base_dir / f
        if f_path.exists():
            shutil.copy2(str(f_path), str(deliv_dir))
            
    # 5. Zip it all
    zip_path = base_dir / 'submission.zip'
    if zip_path.exists():
        zip_path.unlink()
        
    print(f"Creating archive {zip_path}...")
    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(str(deliv_dir)):
            for file in files:
                file_path = os.path.join(root, file)
                # Map deliverables/xxx to Urban_Green_Logistics/xxx
                rel_path = os.path.relpath(file_path, str(deliv_dir))
                arcname = os.path.join('Urban_Green_Logistics', rel_path)
                zipf.write(file_path, arcname)
                
    print("Packaging complete! You can re-run this script anytime after adjusting charts.")

if __name__ == "__main__":
    create_deliverables_package()
