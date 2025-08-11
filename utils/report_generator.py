import os
import csv
import json
from datetime import datetime

def initialize_report(config):
    run_report = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(config['dataset']['output_dir'], config['dataset']['name'], f'generation_report_{timestamp}.csv')
    
    # Initialize CSV file with headers
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)
    with open(report_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'scene_id', 'num_object_classes', 'num_images_planned', 'num_images_generated',
            'total_object_instances', 'object_data'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return report_filename, run_report

def create_scene_report(scene_id, num_object_classes, num_images):
    return {
        'scene_id': scene_id,
        'num_object_classes': num_object_classes,
        'num_images_planned': num_images,
        'num_images_generated': 0,
        'objects': [],
        'object_sizes': {}
    }

def collect_object_data(scene_objects, object_sizes):
    object_instance_counts = {}
    
    print(f"Placing {len(scene_objects)} object instances:")
    
    for j, obj in enumerate(scene_objects):
        obj_path = obj.get_cp("model_path") if obj.has_cp("model_path") else "unknown"
        obj_id = obj.get_cp("obj_id") if obj.has_cp("obj_id") else "unknown"
        instance_id = obj.get_cp("instance_id") if obj.has_cp("instance_id") else "unknown"
        category = obj.get_cp("category") if obj.has_cp("category") else "unknown"
        name = obj.get_cp("model_name") if obj.has_cp("model_name") else "unknown"
        dataset_source = obj.get_cp("dataset_source") if obj.has_cp("dataset_source") else "unknown"
        assigned_size = object_sizes.get(obj_id, "unknown")
        
        # Count instances per object
        if obj_id not in object_instance_counts:
            object_instance_counts[obj_id] = {
                'count': 0,
                'category': category,
                'name': name,
                'dataset_source': dataset_source,
                'model_path': obj_path,
                'size': assigned_size
            }
        object_instance_counts[obj_id]['count'] += 1
        
        print(f"  {j+1:2d}. {obj.get_name()} (obj_id: {obj_id}, cat: {category}, name: {name}, instance: {instance_id}, size: {assigned_size:.3f}m) -> {os.path.basename(obj_path)}")
    
    return object_instance_counts

def write_scene_report(report_filename, scene_report, run_report):
    total_instances = sum(obj_data['count'] for obj_data in scene_report['objects'].values())
    object_data_json = json.dumps(scene_report['objects'], indent=2)
    
    row = {
        'scene_id': scene_report['scene_id'],
        'num_object_classes': scene_report['num_object_classes'],
        'num_images_planned': scene_report['num_images_planned'],
        'num_images_generated': scene_report['num_images_generated'],
        'total_object_instances': total_instances,
        'object_data': object_data_json
    }
    
    # Append to CSV file
    with open(report_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'scene_id', 'num_object_classes', 'num_images_planned', 'num_images_generated',
            'total_object_instances', 'object_data'
        ])
        writer.writerow(row)
    
    # Add to run report
    run_report.append(scene_report)
    print(f"Scene {scene_report['scene_id']} completed: {scene_report['num_images_generated']}/{scene_report['num_images_planned']} images generated")
    print(f"Report updated: {report_filename}")
