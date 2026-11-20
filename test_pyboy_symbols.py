# test_pyboy_symbols.py
import pyboy
import os

# --- Configuration ---
rom_path = "PokemonRed.gb"      # Assumes ROM is in the same directory
sym_path = "pokered.sym"      # Assumes SYM is in the same directory
# ---------------------

# Get PyBoy version correctly (available since PyBoy 1.5.0)
try:
    print(f"PyBoy version: {pyboy.PYBOY_VERSION}")
except AttributeError:
    print("Could not retrieve PyBoy version via pyboy.PYBOY_VERSION (might be older than 1.5.0)")


print(f"Attempting to use ROM Path: {os.path.abspath(rom_path)}")
print(f"Attempting to use Symbol Path: {os.path.abspath(sym_path)}")
print(f"Symbol file exists? {os.path.exists(sym_path)}")

instance = None # Initialize to None

try:
    print("Initializing PyBoy...")
    instance = pyboy.PyBoy(rom_path, symbol_file=sym_path, window_type="null") # Use null window for testing
    print(f"PyBoy object created. ID: {id(instance)}")

    if hasattr(instance, 'symbol'):
        print("[SUCCESS] PyBoy instance HAS attribute 'symbol'.")
        if instance.symbol is not None:
            print("[SUCCESS] instance.symbol is not None.")
            try:
                # Try getting a common symbol
                addr = instance.symbol.get_symbol("wTileMap")
                print(f"[SUCCESS] Got address for wTileMap: {addr:04x}") # Print address in hex
            except Exception as e:
                print(f"[ERROR] Failed to get symbol address even though .symbol exists: {e}")
        else:
            print("[WARNING] instance.symbol exists BUT IS NONE.")
    else:
        print("[ERROR] CRITICAL - PyBoy instance DOES NOT HAVE attribute 'symbol'.")

except Exception as e:
    print(f"[ERROR] An exception occurred during PyBoy initialization or testing: {e}")
    import traceback
    traceback.print_exc()
finally:
    if instance:
        instance.stop() # Clean up PyBoy instance if it was created
        print("PyBoy instance stopped.")