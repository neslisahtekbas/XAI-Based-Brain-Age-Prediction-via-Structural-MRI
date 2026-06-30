import nibabel as nib
import zipfile
import matplotlib.pyplot as plt
import io

# 1. Zip içindeki ilk .nii.gz dosyasını bulalım
zip_path = r'C:\Users\pc\Desktop\Sınavlar\uyz\archive.zip'

with zipfile.ZipFile(zip_path, 'r') as z:
    # Uzantısı .nii.gz olan dosyaları listele
    nii_files = [f for f in z.namelist() if f.endswith('.nii')]
    
    if not nii_files:
        print("Zip içinde .nii.gz dosyası bulunamadı!")
    else:
        # Örnek olarak ilk dosyayı seçelim
        target_file = nii_files[0]
        print(f"Görselleştirilen dosya: {target_file}")
        
        # Dosyayı hafızaya (memory) alalım
        with z.open(target_file) as f:
            file_content = io.BytesIO(f.read())
            # Nibabel ile 3D görüntüyü yükle
            img = nib.FileHolder(fileobj=file_content)
            data = nib.Nifti1Image.from_file_map({'header': img, 'image': img}).get_fdata()

        # 2. Beynin tam ortasından bir kesit alalım (Aksiyel düzlem)
        # data bir 3D dizidir (Örn: 256, 256, 150)
        orta_kesit_no = data.shape[2] // 2
        kesit = data[:, :, orta_kesit_no]

        # 3. Görselleştirme
        plt.figure(figsize=(8, 8))
        # MRI görüntüleri genelde 'gray' (gri tonlamalı) gösterilir
        plt.imshow(kesit.T, cmap='gray', origin='lower')
        plt.title(f"MRI Kesiti (Katman No: {orta_kesit_no})")
        plt.axis('off') # Eksenleri gizle
        plt.show()