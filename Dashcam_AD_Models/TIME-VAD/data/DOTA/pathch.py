#!/usr/bin/env python3
"""
Script to convert Unix/Linux paths in DOTA list files
"""

def convert_path(unix_path):
    """
    Convert Unix path to new Unix path format with special handling for P2RTFM and outDAD
    """
    # Remove leading/trailing whitespace
    unix_path = unix_path.strip()
    
    # Replace the base path
    if unix_path.startswith('/home/sumit/'):
        # Remove '/home/sumit/' and replace with appropriate base path
        relative_path = unix_path[len('/home/sumit/'):]
        
        # Special handling for P2RTFM paths
        if relative_path.startswith('P2RTFM/'):
            new_path = '/media/sumit/040F-55ED/Lab_PC/' + relative_path
        # Special handling for outDAD paths (remove RTFM-main/ if present)
        elif 'outDAD/' in relative_path:
            # Handle both cases: with or without RTFM-main prefix
            if relative_path.startswith('RTFM-main/outDAD/'):
                clean_path = relative_path[len('RTFM-main/'):]  # Remove RTFM-main/
                new_path = '/media/sumit/040F-55ED/Lab_PC/' + clean_path
            else:
                new_path = '/media/sumit/040F-55ED/Lab_PC/' + relative_path
        # Default case for other paths (like DOTA_feat)
        else:
            new_path = '/media/sumit/040F-55ED/Lab_PC/new_map/' + relative_path
        
        return new_path
    else:
        # If path doesn't match expected pattern, return as is
        return unix_path

def main():
    input_file = "train_dota.list"
    output_file = "train_dota.list"  # You can change this to overwrite the original
    
    try:
        # Read the original file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Convert each path
        converted_lines = []
        for line in lines:
            if line.strip():  # Skip empty lines
                converted_path = convert_path(line)
                converted_lines.append(converted_path + '\n')
            else:
                converted_lines.append(line)  # Preserve empty lines
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.writelines(converted_lines)
        
        print(f"Successfully converted {len([l for l in lines if l.strip()])} paths")
        print(f"Output saved to: {output_file}")
        
        # Show a sample of the conversion
        if lines:
            print("\nSample conversion:")
            print(f"Original: {lines[0].strip()}")
            print(f"Converted: {convert_path(lines[0])}")
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
