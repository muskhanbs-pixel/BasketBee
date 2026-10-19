"""
BasketBee - human-in-the-loop approvals.

Agents only ever create PENDING recommendations. A manager approves or rejects them here.
Approving a reorder creates a real purchase order in the (mock) ERP.

    pending  --approve-->  approved  --> purchase order created in ERP
             --reject--->  rejected
"""

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.db.database import SessionLocal
from backend.db.models import AuditLog, Recommendation
from backend.tools.adapters import AdapterError, ERPAdapter


class ApprovalError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def rec_to_dict(rec: Recommendation) -> dict:
    return {
        "id": rec.rec_id, "type": rec.type, "status": rec.status, "created_by": rec.created_by_agent,
        "decided_by": rec.decided_by, "comment": rec.comment,
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
        **json.loads(rec.payload_json),
    }


def list_recommendations(status: str | None = "pending", session_factory: sessionmaker = SessionLocal) -> list[dict]:
    with session_factory() as db:
        q = select(Recommendation).order_by(Recommendation.rec_id.desc())
        if status:
            q = q.where(Recommendation.status == status)
        return [rec_to_dict(r) for r in db.scalars(q)]


def _get_pending(db: Session, rec_id: int) -> Recommendation:
    rec = db.get(Recommendation, rec_id)
    if not rec:
        raise ApprovalError("NOT_FOUND", f"Recommendation {rec_id} does not exist")
    if rec.status != "pending":
        raise ApprovalError("INVALID", f"Recommendation {rec_id} is already {rec.status}")
    return rec


def _audit(db: Session, action: str, rec_id: int, who: str, output: dict | None = None):
    db.add(AuditLog(session_id=f"approvals:{who}", agent="human", tool=action,
                    input_json=json.dumps({"recommendation_id": rec_id}),
                    output_json=json.dumps(output, default=str) if output else None))


def approve(rec_id: int, decided_by: str, comment: str | None = None,
            erp: ERPAdapter | None = None, session_factory: sessionmaker = SessionLocal) -> dict:
    """Approve a reorder -> create the purchase order in the ERP.
    If the ERP is down, the recommendation goes back to 'pending' so nothing is lost."""
    erp = erp or ERPAdapter.in_process()
    with session_factory() as db:
        rec = _get_pending(db, rec_id)
        payload = json.loads(rec.payload_json)
        rec.status, rec.decided_by, rec.comment = "approved", decided_by, comment
        db.commit()   # the ERP only accepts POs for APPROVED recommendations

        try:
            po = erp.create_purchase_order(rec_id, payload["supplier_id"], payload["warehouse_id"],
                                           payload["sku"], payload["qty"], decided_by)
        except AdapterError as e:
            rec.status, rec.decided_by = "pending", None
            _audit(db, "approve_failed", rec_id, decided_by, {"error": e.code, "message": e.message})
            db.commit()
            raise ApprovalError(e.code, f"Could not create the purchase order: {e.message}. "
                                        f"The recommendation is still pending - try again.")
        _audit(db, "approve", rec_id, decided_by, po)
        db.commit()
        return {"recommendation": rec_to_dict(rec), "purchase_order": po}


def reject(rec_id: int, decided_by: str, comment: str | None = None,
           session_factory: sessionmaker = SessionLocal) -> dict:
    with session_factory() as db:
        rec = _get_pending(db, rec_id)
        rec.status, rec.decided_by, rec.comment = "rejected", decided_by, comment
        _audit(db, "reject", rec_id, decided_by, {"comment": comment, "at": datetime.now().isoformat()})
        db.commit()
        return {"recommendation": rec_to_dict(rec)}