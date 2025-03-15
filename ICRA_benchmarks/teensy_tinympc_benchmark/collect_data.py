import serial
import time
import threading
import queue

# Configuration
PORT = '/dev/cu.usbmodem132804901'  # Your Teensy's port
BAUD_RATE = 115200
OUTPUT_FILE = 'benchmark_results_12_86.csv'
BUFFER_SIZE = 1000  # Number of lines to buffer before writing to file

def serial_reader(ser, data_queue, stop_event):
    """Thread function to read from serial port"""
    while not stop_event.is_set():
        if ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8').strip()
                data_queue.put(line)
            except Exception as e:
                print(f"Error reading serial: {e}")
                time.sleep(0.01)
        else:
            # Small sleep to prevent CPU hogging when no data
            time.sleep(0.001)

def file_writer(data_queue, stop_event):
    """Thread function to write data to file"""
    buffer = []
    
    with open(OUTPUT_FILE, 'w') as f:
        while not stop_event.is_set() or not data_queue.empty():
            try:
                # Get data with timeout to allow checking stop_event
                line = data_queue.get(timeout=0.1)
                buffer.append(line)
                
                # Write to file when buffer reaches size or on last data
                if len(buffer) >= BUFFER_SIZE:
                    f.write('\n'.join(buffer) + '\n')
                    f.flush()
                    buffer = []
                    
            except queue.Empty:
                # If we have data in buffer and no new data for a while, write it
                if buffer:
                    f.write('\n'.join(buffer) + '\n')
                    f.flush()
                    buffer = []
            except Exception as e:
                print(f"Error writing to file: {e}")
        
        # Write any remaining data
        if buffer:
            f.write('\n'.join(buffer) + '\n')

def main():
    print(f"Starting data collection on {PORT}...")
    print(f"Press Ctrl+C to stop and save data to {OUTPUT_FILE}")
    
    # Create communication queue and stop event
    data_queue = queue.Queue()
    stop_event = threading.Event()
    
    try:
        # Open serial port
        ser = serial.Serial(port=PORT, baudrate=BAUD_RATE, timeout=1)
        
        # Create and start threads
        reader_thread = threading.Thread(target=serial_reader, args=(ser, data_queue, stop_event))
        writer_thread = threading.Thread(target=file_writer, args=(data_queue, stop_event))
        
        reader_thread.start()
        writer_thread.start()
        
        # Main thread just waits for keyboard interrupt
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping data collection...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Signal threads to stop
        stop_event.set()
        
        # Wait for threads to finish
        if 'reader_thread' in locals() and reader_thread.is_alive():
            reader_thread.join(timeout=2)
        if 'writer_thread' in locals() and writer_thread.is_alive():
            writer_thread.join(timeout=2)
            
        # Close serial port
        if 'ser' in locals():
            ser.close()
            
        print("Data collection complete.")

if __name__ == "__main__":
    main()