import torch

print("==========================================")
print("       CUDA- / GRAFIKKARTEN-TEST          ")
print("==========================================")

# Prüfen, ob CUDA verfügbar ist
cuda_bereit = torch.cuda.is_available()
print(f"GPU-Beschleunigung verfügbar: {cuda_bereit}")

if cuda_bereit:
    print(f"Anzahl verfügbarer GPUs:      {torch.cuda.device_count()}")
    print(f"Name der Grafikkarte:        {torch.cuda.get_device_name(0)}")
    
    # Ein kleiner Live-Test: Erstelle einen Tensor direkt auf der GPU
    x = torch.rand(3, 3).cuda()
    print("\n[Erfolg] Ein Test-Tensor wurde erfolgreich auf deiner GPU berechnet!")
    print(x)
else:
    print("\n[Fehler] PyTorch kann deine Grafikkarte leider nicht sehen.")
    print("Mögliche Gründe: Falsche Python/PyTorch-Version oder veralteter Grafikkartentreiber.")
print("==========================================")