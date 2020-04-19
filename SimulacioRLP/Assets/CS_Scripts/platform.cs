using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class platform : MonoBehaviour
{
    [SerializeField] float angle = 0.1f;
    [SerializeField] float maxRotation = 5;
    Vector3 rotation;
    float rotationBound = 10;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

        rotation = Vector3.zero;
        Vector3 er = transform.rotation.eulerAngles;
        Debug.Log(er);
        Debug.Log((er.z > 360 - maxRotation || er.z <= maxRotation));
        if (Input.GetKey(KeyCode.LeftArrow) && (er.z < maxRotation || er.z >= 360-maxRotation - rotationBound))
        {
            rotation.z = angle;
        }
        if (Input.GetKey(KeyCode.RightArrow) && (er.z > 360-maxRotation || er.z <= maxRotation + rotationBound))
        {
            rotation.z = -angle;
        }
        if (Input.GetKey(KeyCode.UpArrow) && (er.x < maxRotation || er.x >= 360 - maxRotation - rotationBound))
        {
            rotation.x = angle;
        }
        if (Input.GetKey(KeyCode.DownArrow) && (er.x > 360 - maxRotation || er.x <= maxRotation + rotationBound))
        {
            rotation.x = -angle;
        }

        transform.Rotate(rotation);
    }
}
