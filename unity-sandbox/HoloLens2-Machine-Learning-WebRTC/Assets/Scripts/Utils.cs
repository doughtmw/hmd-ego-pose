using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Utils
{
    // Convert from system numerics to unity matrix 4x4
    public static Matrix4x4 Mat4x4FromFloat4x4(System.Numerics.Matrix4x4 m)
    {
        return new Matrix4x4()
        {
            m00 = m.M11,
            m10 = m.M21,
            m20 = m.M31,
            m30 = m.M41,

            m01 = m.M12,
            m11 = m.M22,
            m21 = m.M32,
            m31 = m.M42,

            m02 = m.M13,
            m12 = m.M23,
            m22 = m.M33,
            m32 = m.M43,

            m03 = m.M14,
            m13 = m.M24,
            m23 = m.M34,
            m33 = m.M44,
        };
    }

    public static Matrix4x4 WindowsToUnityMatrix4x4(System.Numerics.Matrix4x4 m)
    {
        return new Matrix4x4(
            new Vector4(m.M11, m.M21, m.M31, m.M41),
            new Vector4(m.M12, m.M22, m.M32, m.M42),
            new Vector4(m.M13, m.M23, m.M33, m.M43),
            new Vector4(m.M14, m.M24, m.M34, m.M44));
    }

    public static Quaternion WindowsVector3ToUnityQuaterion(System.Numerics.Vector3 v)
    {
        return Quaternion.Euler(new Vector3(v.X, v.Y, v.Z));
    }

    /// <summary>
    /// Handle conversion of the incoming transform from c++ component
    /// to the correct configuration for the Unity coordinate system.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="rot"></param>
    /// <returns></returns>
    public static Matrix4x4 GetTransformInUnityCamera(Vector3 pos, Quaternion rot)
    {
        // right-handed coordinates system (OpenCV) to left-handed one (Unity)
        //var t = new Vector3(pos.x, -pos.y, pos.z);
        var t = new Vector3(-pos.x, -pos.y, pos.z);

        // Compose a matrix
        var T = Matrix4x4.TRS(t, rot, Vector3.one);
        T.m20 *= -1.0f;
        T.m21 *= -1.0f;
        T.m22 *= -1.0f;
        T.m23 *= -1.0f;

        return T;
    }

    public static Vector3 GetVectorFromMatrix(Matrix4x4 m)
    {
        return m.GetColumn(3);
    }

    public static Quaternion GetQuatFromMatrix(Matrix4x4 m)
    {
        return Quaternion.LookRotation(m.GetColumn(2), m.GetColumn(1));
    }


    // Get a rotation quaternion from rodrigues
    public static Quaternion RotationQuatFromRodrigues(Vector3 v)
    {
        var angle = Mathf.Rad2Deg * v.magnitude;
        var axis = v.normalized;
        Quaternion q = Quaternion.AngleAxis(angle, axis);

        // Ensure: 
        // Positive x axis is in the left direction of the observed marker
        // Positive y axis is in the upward direction of the observed marker
        // Positive z axis is facing outward from the observed marker
        // Convert from rodrigues to quaternion representation of angle
        q = Quaternion.Euler(
            -1.0f * q.eulerAngles.x,
            q.eulerAngles.y,
            -1.0f * q.eulerAngles.z) * Quaternion.Euler(0, 0, 180);

        return q;
    }
}
