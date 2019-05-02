using UnityEngine;

namespace mlagentsdev.InferenceBrain
{
    public class NNModel : ScriptableObject
    {
        [HideInInspector]
        public byte[] Value;
    }
}
